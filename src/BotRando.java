import java.util.ArrayList;
import java.util.List;


public class BotRando implements ChessAI {
    @Override
    public Move calculateBestMove(ChessGameView game, ChessPiece.Color aiColor) {
        List<Move> allMoves = new ArrayList<>();
        List<Move> captureMoves = new ArrayList<>();

        for (int l = 0; l < game.getLevels(); l++) {
            for (int x = 0; x < game.getGridSize(); x++) {
                for (int z = 0; z < game.getGridSize(); z++) {
                    Position pos = new Position(l, x, z);
                    ChessPiece piece = game.getPieceAt(pos);
                    if (piece != null && piece.color == aiColor) {
                        allMoves.addAll(game.getValidMoves(pos));
                        captureMoves.addAll(game.getCaptureMoves(pos));
                    }
                }
            }
        }

        if (allMoves.isEmpty()) return null;
        if (!captureMoves.isEmpty()) return captureMoves.get((int) (Math.random() * captureMoves.size()));
        return allMoves.get((int) (Math.random() * allMoves.size()));
    }

    @Override
    public String getAIName() {
        return "Random Bot";
    }
}