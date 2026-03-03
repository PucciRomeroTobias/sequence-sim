/**
 * Map from card string (e.g. "10D", "AS", "JH") to image path.
 * The SVG files use "T" for 10 (e.g. TD.svg for 10 of Diamonds).
 */
const CARD_IMG_MAP = {};

const suits = ["S", "H", "D", "C"];
const ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "Q", "K", "A"];

for (const s of suits) {
  for (const r of ranks) {
    CARD_IMG_MAP[`${r}${s}`] = `/cards/${r}${s}.svg`;
  }
  // 10s use "T" prefix in SVG filenames
  CARD_IMG_MAP[`10${s}`] = `/cards/T${s}.svg`;
}

// Jacks - use PNG for better quality
CARD_IMG_MAP["JS"] = "/cards/JS.png";
CARD_IMG_MAP["JH"] = "/cards/JH.png";
CARD_IMG_MAP["JD"] = "/cards/JD.png";
CARD_IMG_MAP["JC"] = "/cards/JC.png";

export default CARD_IMG_MAP;
export const CARD_BACK = "/cards/back.svg";
