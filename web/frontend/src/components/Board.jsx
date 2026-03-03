import { memo } from "react";
import CARD_IMG_MAP, { CARD_BACK } from "../cardImages";

function Cell({ cell, row, col, highlighted, isLastAiMove, onClick, gameOver }) {
  const isFree = cell.card === "FREE";
  const cardImg = isFree ? CARD_BACK : CARD_IMG_MAP[cell.card];

  let chipClass = "";
  if (cell.chip === "blue") chipClass = "chip-blue";
  else if (cell.chip === "red") chipClass = "chip-red";
  else if (cell.chip === "corner") chipClass = "chip-corner";

  const inSeq = cell.inSequence;

  const cellClasses = [
    "board-cell",
    highlighted && "cell-highlighted",
    isLastAiMove && !gameOver && "cell-last-ai",
    inSeq === "blue" && "cell-seq-blue",
    inSeq === "red" && "cell-seq-red",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={cellClasses} onClick={() => onClick(row, col)}>
      {cardImg && <img src={cardImg} alt={cell.card} draggable={false} />}
      {chipClass && <div className={`chip ${chipClass}`} />}
      {isFree && <div className="chip chip-corner" />}
    </div>
  );
}

const MemoCell = memo(Cell);

export default function Board({
  board,
  highlightedCells,
  lastAiMove,
  onCellClick,
  isMyTurn,
  gameOver,
}) {
  const isHighlighted = (r, c) =>
    highlightedCells.some((p) => p[0] === r && p[1] === c);

  const isLastAi = (r, c) =>
    lastAiMove?.position?.[0] === r && lastAiMove?.position?.[1] === c;

  return (
    <div className="board-container">
      <div className="board-grid">
        {board.map((row, r) =>
          row.map((cell, c) => (
            <MemoCell
              key={`${r}-${c}`}
              cell={cell}
              row={r}
              col={c}
              highlighted={isMyTurn && isHighlighted(r, c)}
              isLastAiMove={isLastAi(r, c)}
              onClick={isMyTurn ? onCellClick : () => {}}
              gameOver={gameOver}
            />
          ))
        )}
      </div>
    </div>
  );
}
