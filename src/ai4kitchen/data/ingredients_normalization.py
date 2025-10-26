import re
import unicodedata
from typing import Iterable, List

UNITS = {
    # core volume/weight
    "teaspoon",
    "teaspoons",
    "tsp",
    "tsp.",
    "tablespoon",
    "tablespoons",
    "tbsp",
    "tbsp.",
    "cup",
    "cups",
    "ounce",
    "ounces",
    "oz",
    "pound",
    "pounds",
    "lb",
    "lbs",
    "gram",
    "grams",
    "g",
    "kilogram",
    "kilograms",
    "kg",
    "milliliter",
    "milliliters",
    "ml",
    "liter",
    "liters",
    "l",
    "quart",
    "quarts",
    "qt",
    "pint",
    "pints",
    # frequent "container-ish" tokens in this dataset
    "can",
    "cans",
    "package",
    "packages",
    "jar",
    "jars",
    "stick",
    "sticks",
}

PREP_WORDS = {
    # very frequent modifiers
    "chopped",
    "minced",
    "grated",
    "fresh",
    "freshly",
    "large",
    "small",
    "ground",
    "peeled",
    "thinly",
    "finely",
    "coarsely",
    "sliced",
    "diced",
    "halved",
    "seeded",
    "pitted",
    "optional",
    "plus",
    "more",
    "divided",
    "about",
    "roughly",
    "to",
    "taste",
    "hot",
    "cold",
    "storebought",
    "homemade",
    "softened",
    "melted",
    "shredded",
    "cubed",
    "drained",
    "rinsed",
    "packed",
    "lightly",
    "beaten",
    "juiced"
    # common noise from instructions slipping into ingredients
    "cut",
    "into",
    "pieces",
    "inch",
    "inches",
    "thick",
    "room",
    "temperature",
    "rinsed",
    "drained"
    # extra common ones (comment out if you want it ultra-minimal)
    ,
    "and",
    "or",
    "of",
    "a",
    "the",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
}

# keep a short list of *useful* multi-word ingredients that appear a lot
MULTIWORD_INGS = {
    "olive oil",
    "vegetable oil",
    "sesame oil",
    "coconut milk",
    "kosher salt",
    "black pepper",
    "brown sugar",
    "baking soda",
    "baking powder",
    "soy sauce",
    "fish sauce",
    "hot sauce",
    "lemon juice",
    "lime juice",
    "white wine",
    "red wine",
    "apple cider",
    "cider vinegar",
    "apple cider vinegar",
    "heavy cream",
    "sour cream",
    "cream cheese",
    "chicken broth",
    "chicken stock",
    "beef broth",
    "vegetable broth",
}

# US ↔ UK synonym mapping (expand minimally)
SYNONYMS = {
    "cilantro": "coriander",
    "arugula": "rocket",
    # a few extra common ones (comment out if you want it ultra-minimal)
    "zucchini": "courgette",
    "eggplant": "aubergine",
    "scallion": "spring onion",
    "scallions": "spring onion",
    "powdered sugar": "icing sugar",
}

# ---------- helpers ----------


class IngredientNormalizer:

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )

    @staticmethod
    def _join_multiwords(text: str, multiwords: Iterable[str]) -> str:
        """Join known multiword ingredients with underscores *before* tokenization."""
        for mw in sorted(multiwords, key=len, reverse=True):
            # surround with spaces to avoid partial matches, then normalize spacing afterwards
            pattern = r"\b" + re.escape(mw) + r"\b"
            text = re.sub(pattern, mw.replace(" ", "_"), text)
        return text

    @staticmethod
    def _remove_quantities(text: str) -> str:
        """Remove numeric quantities (including mixed fractions like 1-1/2)."""
        _FRACTION_RE = re.compile(
            r"\b\d+([/-]\d+)?\b|\b\d+\.\d+\b"
        )  # 1, 1/2, 1-1/2, 1.5, etc.
        return _FRACTION_RE.sub(" ", text)

    @staticmethod
    def _remove_parentheticals_and_punctuation(text: str) -> str:
        """Substitutes parentheses and punctuation with spaces."""
        text = re.sub(r"[()\[\],;:/\-–—]", " ", text)
        return text

    @staticmethod
    def _apply_synonyms(text: str, synonyms: dict) -> str:
        """Map phrase-level synonyms first (handles e.g., 'powdered sugar')."""
        # Do longer (multiword) keys first to avoid partial overwrites
        for k in sorted(synonyms, key=lambda s: len(s), reverse=True):
            v = synonyms[k]
            text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        _WORD_RE = re.compile(
            r"[a-z]+(?:'[a-z]+)?"
        )  # simple word matcher (after lowercase+accent-strip)
        return _WORD_RE.findall(text)

    @staticmethod
    def _drop_clutter(text: str) -> List[str]:
        toks = IngredientNormalizer._tokenize(text)
        # Remove units and prep words from the text.
        toks = [t for t in toks if t not in UNITS and t not in PREP_WORDS]
        # collapse duplicate underscores if any, and discard pure underscores
        toks = [re.sub(r"_+", "_", t).strip("_") for t in toks]
        toks = [t for t in toks if t]
        return toks

    @staticmethod
    def normalize_ingredient(s: str) -> List[str]:
        """
        Normalize one *ingredient line* (which may include qty, unit, prep text).
        Returns a list of normalized tokens/phrases (underscored if multiword).
        """
        # 1) lowercase + strip accents
        s = IngredientNormalizer._strip_accents(s.lower())

        # 2) quick punctuation -> spaces
        s = IngredientNormalizer._remove_parentheticals_and_punctuation(s)

        # 3) remove numeric quantities (including mixed fractions like 1-1/2)
        s = IngredientNormalizer._remove_quantities(s)

        # 4) map synonyms *before* we join multiwords (to allow phrase-level replacements)
        s = IngredientNormalizer._apply_synonyms(s, SYNONYMS)

        # 5) join multi-word ingredients so they stay intact
        s = IngredientNormalizer._join_multiwords(s, MULTIWORD_INGS)

        # 7) drop units and prep words, collapse duplicate underscores if any, and discard pure underscores
        toks = IngredientNormalizer._drop_clutter(s)

        return toks

    def normalize_ingredient_list(ings: Iterable[str]) -> List[str]:
        """
        Normalize a list of ingredient strings and return a flat list of normalized items.
        You can also choose to deduplicate while preserving order.
        """
        out: List[str] = []
        seen = set()
        for line in ings:
            for tok in IngredientNormalizer.normalize_ingredient(line):
                if tok not in seen:
                    seen.add(tok)
                    out.append(tok)
        return out


# ---------- example usage on your dataframe ----------

if __name__ == "__main__":
    import ast
    import pandas as pd
    from ai4kitchen.data.ingredients_normalization import IngredientNormalizer

    df = pd.read_csv(
        "~/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    )

    def parse_list(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

    normalized = []
    for s in df["Cleaned_Ingredients"].fillna("[]"):
        ings = parse_list(s)
        normalized.append(IngredientNormalizer.normalize_ingredient_list(ings))

    df["normalized_ingredients"] = normalized

    # peek
    print(df[["Title", "normalized_ingredients"]].head(5).to_string(index=False))
