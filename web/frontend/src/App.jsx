import { useState, useEffect, useCallback } from "react";
import { io } from "socket.io-client";
import Board from "./components/Board";
import PlayerHand from "./components/PlayerHand";
import GameInfo from "./components/GameInfo";

const socket = io("/", { transports: ["websocket", "polling"] });

export default function App() {
  const [gameState, setGameState] = useState(null);
  const [selectedCard, setSelectedCard] = useState(null);
  const [highlightedCells, setHighlightedCells] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    socket.on("connect", () => {
      setConnected(true);
      socket.emit("new_game");
    });
    socket.on("disconnect", () => setConnected(false));
    socket.on("gameState", (state) => {
      setGameState(state);
      setSelectedCard(null);
      setHighlightedCells([]);
    });
    socket.on("error", (err) => {
      console.error("Server error:", err.message);
    });

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("gameState");
      socket.off("error");
    };
  }, []);

  const handleCardSelect = useCallback(
    (cardStr) => {
      if (!gameState || gameState.currentTeam !== gameState.humanTeam) return;
      if (!gameState.winner) {
        setSelectedCard(cardStr === selectedCard ? null : cardStr);
        // Find legal positions for this card
        const positions = gameState.legalActions
          .filter((a) => a.card === cardStr && a.position)
          .map((a) => a.position);
        setHighlightedCells(cardStr === selectedCard ? [] : positions);
      }
    },
    [gameState, selectedCard]
  );

  const handleCellClick = useCallback(
    (row, col) => {
      if (!gameState || !selectedCard) return;
      if (gameState.currentTeam !== gameState.humanTeam) return;
      if (gameState.winner) return;

      // Find matching legal action
      const action = gameState.legalActions.find(
        (a) =>
          a.card === selectedCard &&
          a.position &&
          a.position[0] === row &&
          a.position[1] === col
      );
      if (action) {
        socket.emit("play_action", action);
      }
    },
    [gameState, selectedCard]
  );

  const handleDiscard = useCallback(
    (cardStr) => {
      if (!gameState) return;
      const action = gameState.legalActions.find(
        (a) => a.card === cardStr && a.type === "dead_card_discard"
      );
      if (action) {
        socket.emit("play_action", action);
      }
    },
    [gameState]
  );

  const handleNewGame = useCallback(() => {
    socket.emit("new_game");
  }, []);

  if (!connected) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white text-xl">
        Connecting to server...
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white text-xl">
        Starting game...
      </div>
    );
  }

  const isMyTurn =
    gameState.currentTeam === gameState.humanTeam && !gameState.winner;

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center py-6 gap-4">
      <h1 className="text-2xl font-bold text-white tracking-wide">
        Sequence
      </h1>

      <GameInfo
        scores={gameState.scores}
        currentTeam={gameState.currentTeam}
        humanTeam={gameState.humanTeam}
        winner={gameState.winner}
        turnNumber={gameState.turnNumber}
        deckCount={gameState.deckCount}
        lastAiMove={gameState.lastAiMove}
        onNewGame={handleNewGame}
      />

      <Board
        board={gameState.board}
        highlightedCells={highlightedCells}
        lastAiMove={gameState.lastAiMove}
        onCellClick={handleCellClick}
        isMyTurn={isMyTurn}
        gameOver={!!gameState.winner}
      />

      <PlayerHand
        hand={gameState.hand}
        selectedCard={selectedCard}
        legalActions={gameState.legalActions}
        isMyTurn={isMyTurn}
        onCardSelect={handleCardSelect}
        onDiscard={handleDiscard}
      />
    </div>
  );
}
