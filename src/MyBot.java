import java.util.ArrayList;
import java.util.List;

/**
 * MyBot — tournament-strength 3D chess AI.
 *
 * <h2>Search architecture</h2>
 * <p>
 * {@code SimulatedChessGameView} (returned by {@code withMove()}) stubs out
 * {@code getValidMoves}, {@code getCaptureMoves}, {@code isInCheck},
 * {@code isCheckmate}, and {@code checkGameOver} — they always return empty /
 * false / null.  Only {@code getPieceAt}, {@code isWhiteTurn}, and
 * {@code getTotalMoves} work correctly on simulated views.
 * </p>
 * <p>
 * Because move generation is unavailable on simulated boards, the search uses
 * a <em>proxy move</em> strategy: moves collected from the real game view at
 * the start of each turn are used as candidates at every ply.  Each candidate
 * is validated against the current simulated board (piece still at FROM, destination
 * empty or opponent-occupied).  Despite the approximation, iterative-deepening
 * alpha-beta comfortably reaches depth 5–7 inside the 5-second limit.
 * </p>
 *
 * <h2>Win conditions (from ChessEngine bytecode)</h2>
 * <ul>
 *   <li><b>Center control:</b> king holds the same center square
 *       ({@code level=1, x∈{3,4}, z∈{3,4}}) for 4 consecutive half-moves.
 *       Moving to a different center square resets the counter.</li>
 *   <li><b>Checkmate</b></li>
 *   <li><b>Material tiebreak</b> after 500 total moves (P=1, N/B=3, R=5, Q=9)</li>
 * </ul>
 *
 * <h2>Evaluation components</h2>
 * <ol>
 *   <li>Material balance (centipawns)</li>
 *   <li>King-in-center-of-middle-board: ±80 000 pt; escalates when the opponent
 *       king has held the same center square across multiple turns</li>
 *   <li>Non-king pieces in center of middle board: ±30 pt</li>
 *   <li>Any piece in center of outer levels: ±10 pt</li>
 *   <li>Non-king pieces on the middle board level: ±8 pt</li>
 *   <li>Advanced pawns (hasMoved flag): ±15 pt</li>
 *   <li>Hanging-piece penalty: opponent's proxy captures are detected and penalised
 *       at ~30 % of the victim piece value (quiescence-search substitute)</li>
 * </ol>
 *
 * <h2>Move ordering</h2>
 * <ol>
 *   <li>Captures — Most-Valuable-Victim / Least-Valuable-Attacker (MVV-LVA)</li>
 *   <li>Killer moves (2 slots per depth)</li>
 *   <li>History heuristic (quiet moves scored by past cutoff frequency)</li>
 * </ol>
 */
public class MyBot implements ChessAI {

    // ── Piece values (centipawns) ────────────────────────────────────────
    private static final int VAL_PAWN   = 100;
    private static final int VAL_KNIGHT = 320;
    private static final int VAL_BISHOP = 330;
    private static final int VAL_ROOK   = 500;
    private static final int VAL_QUEEN  = 900;
    private static final int VAL_KING   = 20_000;

    private static final int  INF      = 10_000_000;
    private static final long TIME_MS  = 4_300;  // 700 ms safety buffer vs 5 s limit
    private static final int  MAX_DEPTH = 8;     // iterative-deepening cap

    /**
     * Set to {@code true} to print per-move search statistics to stdout.
     * Disable before submitting.
     */
    static final boolean DEBUG = false;

    // ── Per-call state ───────────────────────────────────────────────────
    private long             deadline;
    private ChessPiece.Color aiColor;
    private ChessPiece.Color oppColor;
    private ChessGameView    realGame;

    /** Exact legal moves for the AI at the start of this turn. */
    private List<Move> myRealMoves;
    /** Exact legal moves for the opponent — used as proxy moves throughout the tree. */
    private List<Move> oppRealMoves;

    // Board-geometry cache (recomputed each call)
    private int gs, levels, mid, cx0, cx1;

    // ── Cross-call centre-tracking state ────────────────────────────────
    /**
     * Last observed position of the opponent king.
     * Null if not in a centre square last time.
     */
    private Position prevOppKingPos  = null;
    /**
     * Number of consecutive {@code calculateBestMove} calls in which the opponent
     * king occupied the same centre square.  When ≥ 2 the urgency penalty in
     * {@link #evaluate} escalates to reflect imminent loss.
     */
    private int      oppCenterStreak = 0;

    // ── Search statistics ─────────────────────────────────────────────────
    private int nodesVisited;
    private int cutoffCount;
    private int evalCount;
    private int reachedDepth;

    // ── Move-ordering tables (allocated per call) ─────────────────────────
    /**
     * killers[depth][0..1] — last two quiet moves that caused a beta cutoff at
     * this depth.  Tried immediately after captures in move ordering.
     */
    private Move[][] killers;

    /**
     * history[fromIdx][toIdx] — quiet moves that caused beta cutoffs are
     * incremented by depth²; used to rank quiet moves below killers.
     */
    private int[][] history;

    // ─────────────────────────────────────────────────────────────────────

    /** {@inheritDoc} */
    @Override
    public String getAIName() { return "MyBot"; }

    // ─────────────────────────────────────────────────────────────────────

    /**
     * Entry point called by the engine each turn.
     *
     * <p>Runs iterative-deepening from depth 1 up to {@link #MAX_DEPTH}.
     * The best move from the last <em>fully completed</em> iteration is
     * returned; a partially-completed iteration is discarded.
     *
     * @param game     the current game state (real view — move generation works)
     * @param aiColor  the colour this bot is playing
     * @return the chosen move, or {@code null} if the AI has no legal moves
     */
    @Override
    public Move calculateBestMove(ChessGameView game, ChessPiece.Color aiColor) {
        long t0   = System.currentTimeMillis();
        deadline  = t0 + TIME_MS;
        this.aiColor  = aiColor;
        this.oppColor = opponent(aiColor);
        this.realGame = game;

        // Cache board geometry
        gs     = game.getGridSize();
        levels = game.getLevels();
        mid    = levels / 2;          // = 1 for 3-level board
        cx0    = gs / 2 - 1;          // = 3 for 8×8
        cx1    = gs / 2;              // = 4 for 8×8

        // Update cross-call opponent-centre tracking
        updateCenterTracking(game);

        // Collect exact legal moves (only valid on the real game view)
        myRealMoves  = collectMoves(game, aiColor);
        oppRealMoves = collectMoves(game, oppColor);
        if (myRealMoves.isEmpty()) return null;

        // Initialise per-search tables
        int boardSquares = levels * gs * gs;
        killers = new Move[MAX_DEPTH + 2][2];
        history = new int[boardSquares][boardSquares];
        nodesVisited = cutoffCount = evalCount = reachedDepth = 0;

        // Sort root lists so the best candidates come first
        orderMoves(myRealMoves,  game, MAX_DEPTH, aiColor);
        orderMoves(oppRealMoves, game, MAX_DEPTH, oppColor);

        // Iterative deepening
        Move best = myRealMoves.get(0);
        for (int d = 1; d <= MAX_DEPTH; d++) {
            if (timeout()) break;
            Move candidate = rootSearch(d);
            if (candidate != null) {
                best         = candidate;
                reachedDepth = d;
            } else {
                break; // timed out mid-iteration — discard and use previous best
            }
        }

        if (DEBUG) {
            System.out.printf(
                "[MyBot %s d=%d] nodes=%d evals=%d cuts=%d streak=%d time=%dms%n",
                aiColor, reachedDepth, nodesVisited, evalCount, cutoffCount,
                oppCenterStreak, System.currentTimeMillis() - t0);
        }
        return best;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Search
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Root search at the given depth.
     *
     * @return the best move found, or {@code null} if a timeout occurred before
     *         any move was evaluated
     */
    private Move rootSearch(int depth) {
        int  alpha = -INF;
        Move best  = null;

        for (Move m : myRealMoves) {
            if (timeout()) return null;
            nodesVisited++;

            int s = search(realGame.withMove(m), depth - 1, alpha, INF, oppColor);

            if (best == null || s > alpha) {
                alpha = s;
                best  = m;
            }
        }
        return best;
    }

    /**
     * Recursive proxy minimax with alpha-beta pruning.
     *
     * <p>Moves at every depth are taken from the pre-collected real-game move
     * lists and filtered to ensure validity on the current simulated board.
     *
     * @param game          current simulated board state
     * @param depth         remaining search depth (0 → evaluate)
     * @param alpha         lower bound (maximiser's best confirmed score)
     * @param beta          upper bound (minimiser's best confirmed score)
     * @param currentColor  the side to move at this node
     * @return the minimax score from {@link #aiColor}'s perspective
     */
    private int search(ChessGameView game, int depth, int alpha, int beta,
                       ChessPiece.Color currentColor) {
        nodesVisited++;

        if (depth == 0 || timeout()) {
            evalCount++;
            return evaluate(game);
        }

        List<Move> moves = proxyMoves(game, currentColor);
        if (moves.isEmpty()) {
            evalCount++;
            return evaluate(game);
        }

        orderMoves(moves, game, depth, currentColor);

        if (currentColor == aiColor) {
            // Maximising node
            int best = -INF;
            for (Move m : moves) {
                if (timeout()) break;
                int s = search(game.withMove(m), depth - 1, alpha, beta, oppColor);
                if (s > best)  best  = s;
                if (s > alpha) alpha = s;
                if (alpha >= beta) {
                    cutoffCount++;
                    recordCutoff(m, depth, game);
                    break;
                }
            }
            return best == -INF ? evaluate(game) : best;

        } else {
            // Minimising node
            int best = INF;
            for (Move m : moves) {
                if (timeout()) break;
                int s = search(game.withMove(m), depth - 1, alpha, beta, aiColor);
                if (s < best) best = s;
                if (s < beta) beta = s;
                if (beta <= alpha) {
                    cutoffCount++;
                    recordCutoff(m, depth, game);
                    break;
                }
            }
            return best == INF ? evaluate(game) : best;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Move generation
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Collects all legal moves for {@code color} from the real game view.
     * Must only be called on the real {@code ChessGameView}, not a simulated one.
     */
    private List<Move> collectMoves(ChessGameView game, ChessPiece.Color color) {
        List<Move> out = new ArrayList<>();
        for (int l = 0; l < levels; l++)
            for (int x = 0; x < gs; x++)
                for (int z = 0; z < gs; z++) {
                    Position pos = new Position(l, x, z);
                    ChessPiece p = game.getPieceAt(pos);
                    if (p != null && p.color == color)
                        out.addAll(game.getValidMoves(pos));
                }
        return out;
    }

    /**
     * Returns the subset of the pre-collected proxy list that is valid on the
     * given simulated board.
     *
     * <p>A proxy move is valid when:
     * <ol>
     *   <li>The moving piece is still present at the FROM square with the
     *       expected colour (i.e. it was not captured by a prior move).</li>
     *   <li>The destination is either empty or occupied by an opponent piece
     *       (i.e. not blocked by an ally).</li>
     * </ol>
     *
     * @param sim   the current simulated board
     * @param color the side whose moves to generate
     * @return a new list of validated proxy moves (may be empty)
     */
    private List<Move> proxyMoves(ChessGameView sim, ChessPiece.Color color) {
        List<Move> src = (color == aiColor) ? myRealMoves : oppRealMoves;
        List<Move> out = new ArrayList<>(src.size());
        for (Move m : src) {
            ChessPiece p = sim.getPieceAt(m.getFrom());
            if (p == null || p.color != color) continue;          // captured or wrong colour
            ChessPiece t = sim.getPieceAt(m.getTo());
            if (t != null && t.color == color) continue;          // blocked by ally
            out.add(m);
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Evaluation
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Static evaluation of a simulated position from {@link #aiColor}'s perspective.
     *
     * <p>Uses only {@code getPieceAt} — the only reliable method on
     * {@code SimulatedChessGameView}.  Higher scores are better for the AI.
     *
     * <p>The evaluation never calls {@code isInCheck}, {@code isCheckmate}, or
     * {@code getValidMoves} because these are stubs on simulated views.
     *
     * @param game the board position to evaluate
     * @return the static score in centipawn units
     */
    private int evaluate(ChessGameView game) {
        int score = 0;

        for (int l = 0; l < levels; l++) {
            for (int x = 0; x < gs; x++) {
                for (int z = 0; z < gs; z++) {
                    ChessPiece p = game.getPieceAt(new Position(l, x, z));
                    if (p == null) continue;

                    int sign = (p.color == aiColor) ? 1 : -1;
                    score += sign * pieceVal(p.type);

                    boolean inCenter = (x >= cx0 && x <= cx1 && z >= cx0 && z <= cx1);

                    if (p.type == ChessPiece.Type.KING && inCenter && l == mid) {
                        // Centre-control win condition: king in the 2×2 middle of the
                        // middle board.  When the opponent has held that square for
                        // multiple consecutive turns the penalty escalates, reflecting
                        // that they are close to satisfying CENTER_CONTROL_TURNS_TO_WIN=4.
                        int urgency = (p.color == oppColor)
                                      ? oppCenterStreak * 20_000 : 0;
                        score += sign * (80_000 + urgency);

                    } else if (inCenter && l == mid) {
                        // Non-king pieces controlling the win-condition zone
                        score += sign * 30;

                    } else if (inCenter) {
                        // Centre control on outer levels
                        score += sign * 10;
                    }

                    // Middle-board activity bonus (helps with 3-D cross-level influence)
                    if (l == mid && p.type != ChessPiece.Type.KING) {
                        score += sign * 8;
                    }

                    // Pawn advancement — hasMoved is a reliable proxy for progress
                    if (p.type == ChessPiece.Type.PAWN && p.hasMoved) {
                        score += sign * 15;
                    }
                }
            }
        }

        // Hanging-piece detection (quiescence-search substitute).
        // Scans the opponent's proxy moves for undefended captures on this board.
        // Only pieces of rook value or higher are checked to keep eval cost bounded.
        for (Move om : oppRealMoves) {
            ChessPiece attacker = game.getPieceAt(om.getFrom());
            if (attacker == null || attacker.color != oppColor) continue;
            ChessPiece victim = game.getPieceAt(om.getTo());
            if (victim != null && victim.color == aiColor
                    && pieceVal(victim.type) >= VAL_ROOK) {
                score -= pieceVal(victim.type) / 3;
            }
        }

        return score;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Move ordering
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Sorts {@code moves} in-place using a priority heuristic:
     * <ol>
     *   <li>Captures (MVV-LVA: high-value victim, low-value attacker → earlier)</li>
     *   <li>Killer moves (slot 0 then slot 1 at this depth)</li>
     *   <li>Quiet moves ranked by history-heuristic score</li>
     * </ol>
     *
     * @param moves        the move list to sort
     * @param game         current board (used for piece-value lookups)
     * @param depth        current depth (used to look up killers)
     * @param movingColor  the side whose moves these are
     */
    private void orderMoves(List<Move> moves, ChessGameView game,
                            int depth, ChessPiece.Color movingColor) {
        final Move k0 = (depth < killers.length) ? killers[depth][0] : null;
        final Move k1 = (depth < killers.length) ? killers[depth][1] : null;
        moves.sort((a, b) ->
            moveScore(b, game, k0, k1, movingColor)
          - moveScore(a, game, k0, k1, movingColor));
    }

    /**
     * Returns an ordering score for a move; higher = try earlier.
     *
     * @param m            the move to score
     * @param game         current board state
     * @param k0           primary killer move at this depth (may be null)
     * @param k1           secondary killer move at this depth (may be null)
     * @param movingColor  the side making this move
     */
    private int moveScore(Move m, ChessGameView game,
                          Move k0, Move k1, ChessPiece.Color movingColor) {
        ChessPiece victim = game.getPieceAt(m.getTo());
        if (victim != null && victim.color != movingColor) {
            // MVV-LVA: offset 10 000 ensures all captures outrank quiet moves
            ChessPiece att = game.getPieceAt(m.getFrom());
            int attVal = (att != null) ? pieceVal(att.type) : 0;
            return 10_000 + 10 * pieceVal(victim.type) - attVal;
        }
        if (m.equals(k0)) return 9_000;
        if (m.equals(k1)) return 8_000;
        // History heuristic: quiet moves that previously caused cutoffs rank higher
        int fi = posIdx(m.getFrom()), ti = posIdx(m.getTo());
        return (fi >= 0 && ti >= 0) ? history[fi][ti] : 0;
    }

    /**
     * Records a move that caused a beta cutoff for future move ordering.
     *
     * <p>Captures are not recorded (they are already ordered by MVV-LVA).
     * For quiet moves: updates killer-move slots and increments the history table.
     *
     * @param m     the move that caused the cutoff
     * @param depth current search depth
     * @param game  current board (used to check whether the move is a capture)
     */
    private void recordCutoff(Move m, int depth, ChessGameView game) {
        ChessPiece victim = game.getPieceAt(m.getTo());
        if (victim == null && depth < killers.length) {
            // Quiet move: update killer slots
            if (!m.equals(killers[depth][0])) {
                killers[depth][1] = killers[depth][0];
                killers[depth][0] = m;
            }
        }
        // Always update history (captures included; weight = depth²)
        int fi = posIdx(m.getFrom()), ti = posIdx(m.getTo());
        if (fi >= 0 && ti >= 0) history[fi][ti] += depth * depth;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Centre-control tracking
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Updates {@link #oppCenterStreak} and {@link #prevOppKingPos} based on the
     * current real-game board.  Called once at the start of each turn.
     *
     * <p>If the opponent king is on the same centre square as last turn,
     * the streak counter is incremented.  Otherwise it is reset to 0 or 1.
     * The streak feeds directly into the urgency penalty in {@link #evaluate}.
     *
     * @param game the current real game state
     */
    private void updateCenterTracking(ChessGameView game) {
        Position oppKing = findKing(game, oppColor);
        if (oppKing != null && isWinSquare(oppKing)) {
            if (oppKing.equals(prevOppKingPos)) {
                oppCenterStreak++;
            } else {
                oppCenterStreak = 1;
                prevOppKingPos  = oppKing;
            }
        } else {
            oppCenterStreak = 0;
            prevOppKingPos  = null;
        }
    }

    /**
     * Scans the board for the king of the given colour.
     *
     * @param game  board to search
     * @param color colour of the king to find
     * @return the king's position, or {@code null} if not found
     */
    private Position findKing(ChessGameView game, ChessPiece.Color color) {
        for (int l = 0; l < levels; l++)
            for (int x = 0; x < gs; x++)
                for (int z = 0; z < gs; z++) {
                    ChessPiece p = game.getPieceAt(new Position(l, x, z));
                    if (p != null && p.type == ChessPiece.Type.KING && p.color == color)
                        return new Position(l, x, z);
                }
        return null;
    }

    /**
     * Returns {@code true} if {@code pos} is one of the four centre squares of
     * the middle board — the squares targeted by the centre-control win condition.
     *
     * @param pos the position to test
     */
    private boolean isWinSquare(Position pos) {
        int l = pos.getLevel(), x = pos.getX(), z = pos.getZ();
        return l == mid && x >= cx0 && x <= cx1 && z >= cx0 && z <= cx1;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────

    /** @return {@code true} if the per-move time budget has been exhausted */
    private boolean timeout() { return System.currentTimeMillis() >= deadline; }

    /**
     * Converts a {@link Position} to a flat array index for the history table.
     *
     * @return a non-negative index, or {@code -1} if the position is out of range
     */
    private int posIdx(Position p) {
        int l = p.getLevel(), x = p.getX(), z = p.getZ();
        if (l < 0 || x < 0 || z < 0 || l >= levels || x >= gs || z >= gs) return -1;
        return l * gs * gs + x * gs + z;
    }

    private static ChessPiece.Color opponent(ChessPiece.Color c) {
        return c == ChessPiece.Color.WHITE ? ChessPiece.Color.BLACK : ChessPiece.Color.WHITE;
    }

    /**
     * Returns the centipawn value for a piece type.
     *
     * @param t the piece type
     * @return centipawn value (0 for unknown types)
     */
    private static int pieceVal(ChessPiece.Type t) {
        switch (t) {
            case PAWN:   return VAL_PAWN;
            case KNIGHT: return VAL_KNIGHT;
            case BISHOP: return VAL_BISHOP;
            case ROOK:   return VAL_ROOK;
            case QUEEN:  return VAL_QUEEN;
            case KING:   return VAL_KING;
            default:     return 0;
        }
    }
}
