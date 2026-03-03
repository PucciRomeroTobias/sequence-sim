import CARD_IMG_MAP, { CARD_BACK } from "../cardImages";

export default function PlayerHand({
  hand,
  selectedCard,
  legalActions,
  isMyTurn,
  onCardSelect,
  onDiscard,
}) {
  // Check which cards can be discarded (dead cards)
  const discardable = new Set(
    legalActions
      .filter((a) => a.type === "dead_card_discard")
      .map((a) => a.card)
  );

  // Check which cards have playable positions
  const playable = new Set(
    legalActions.filter((a) => a.position).map((a) => a.card)
  );

  return (
    <div className="hand-container">
      <div className="hand-label">Your Hand</div>
      <div className="hand-cards">
        {hand.map((cardStr, i) => {
          const isSelected = selectedCard === cardStr;
          const canPlay = playable.has(cardStr);
          const canDiscard = discardable.has(cardStr);
          const isDead = canDiscard && !canPlay;

          return (
            <div
              key={`${cardStr}-${i}`}
              className={[
                "hand-card",
                isSelected && "hand-card-selected",
                !isMyTurn && "hand-card-disabled",
                isDead && isMyTurn && "hand-card-dead",
              ]
                .filter(Boolean)
                .join(" ")}
              onClick={() => {
                if (!isMyTurn) return;
                if (isDead) {
                  onDiscard(cardStr);
                } else if (canPlay) {
                  onCardSelect(cardStr);
                }
              }}
            >
              <img
                src={CARD_IMG_MAP[cardStr] || CARD_BACK}
                alt={cardStr}
                draggable={false}
              />
              {isDead && isMyTurn && (
                <div className="dead-badge">Discard</div>
              )}
            </div>
          );
        })}
      </div>
      {!isMyTurn && !discardable.size && (
        <div className="text-gray-400 text-sm mt-1">AI is thinking...</div>
      )}
    </div>
  );
}
