export default function GameInfo({
  scores,
  currentTeam,
  humanTeam,
  winner,
  turnNumber,
  deckCount,
  lastAiMove,
  onNewGame,
}) {
  const isMyTurn = currentTeam === humanTeam && !winner;
  const isAiTurn = currentTeam !== humanTeam && !winner;

  return (
    <div className="game-info">
      <div className="scores">
        <div
          className={`score-box score-blue ${
            isMyTurn ? "score-active pulse-blue" : ""
          }`}
        >
          <div className="score-label">You (Blue)</div>
          <div className="score-value">{scores.blue}/2</div>
          {isMyTurn && <div className="turn-indicator turn-indicator-blue">Your turn</div>}
        </div>
        <div className="score-middle">
          {winner ? (
            <div className="winner-text">
              {winner === humanTeam ? "You win!" : "AI wins!"}
              <button onClick={onNewGame} className="new-game-btn">
                Play Again
              </button>
            </div>
          ) : (
            <div className="turn-info">
              <div>Turn {turnNumber + 1}</div>
              <div className="text-xs text-gray-400">
                {deckCount} cards in deck
              </div>
              {!winner && (
                <button onClick={onNewGame} className="reset-btn">
                  Restart
                </button>
              )}
            </div>
          )}
        </div>
        <div
          className={`score-box score-red ${
            isAiTurn ? "score-active pulse-red" : ""
          }`}
        >
          <div className="score-label">AI (Red)</div>
          <div className="score-value">{scores.red}/2</div>
          {isAiTurn && (
            <div className="turn-indicator turn-indicator-red">
              <span className="thinking-dots">Thinking</span>
            </div>
          )}
        </div>
      </div>
      {lastAiMove && (
        <div className="last-ai-move">
          AI played: {lastAiMove.card}
          {lastAiMove.position
            ? ` at (${lastAiMove.position[0]},${lastAiMove.position[1]})`
            : " (discard)"}
        </div>
      )}
    </div>
  );
}
