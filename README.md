# IKEA Intelligent Room Planning Agent

Regelbaserad agent som tar:
- rumsmått och fasta hinder,
- behov (t.ex. `home office`, `guest bed`),
- budget i SEK,
- produktdataset (IKEA-liknande schema),

och returnerar:
- **strukturerad JSON-plan** (valda produkter + placering),
- **deterministisk validering** mot hårda regler,
- **iterativa justeringar** vid konflikter,
- **mänskligt läsbar sammanfattning** med trade-offs.

## Snabbstart

```bash
python ikea_room_planning_agent.py examples/input_sample.json
```

## Inputschema (sammanfattning)

```json
{
  "room": {
    "width_m": 3.2,
    "length_m": 4.0,
    "door": {"x_m": 0.2, "y_m": 0.0, "width_m": 0.9, "swing_direction": "inward_left"},
    "windows": [{"x_m": 1.0, "y_m": 4.0, "width_m": 1.2, "height_m": 1.1}],
    "fixed_features": [{"x_m": 2.6, "y_m": 0.0, "width_m": 0.6, "length_m": 0.6, "description": "radiator"}]
  },
  "needs": ["home office", "guest bed"],
  "budget_SEK": 11000,
  "style": ["Scandinavian", "Minimal"],
  "clearance_preferences": {"walkway_min_m": 0.6},
  "placement_hints": {"prefer_desk_under_window": true},
  "dataset": []
}
```

## Krav på dataset

Varje produkt ska ha:
- `product_id` (sträng)
- `name` (sträng)
- `category` (t.ex. bed, sofa, desk, chair, storage, lighting)
- `width_m`, `length_m`, `height_m` (float)
- `price_SEK` (float)
- `style_tags` (lista strängar)
- `convertible` (bool)
- `notes` (valfri)
- `sku_url` (valfri)

## Output

Agenten returnerar JSON med:
- `status`: `success` eller `best_effort`
- `proposal.items`: placerade produkter (x/y/orientering)
- `validation.failures`: lista över regelbrott med `rule_id` + `rejection_reason`
- `tradeoffs` och `remaining_risks`
- `summary_text`: kort, mänsklig sammanfattning

## Regler som valideras

- R1_ROOM_BOUNDS
- R2_WALKWAY_CLEARANCE
- R3_DESK_LEGROOM
- R4_BED_SIDE_CLEARANCE
- R5_SOFA_FRONT_CLEARANCE
- R6_DOOR_SWING_ACCESS
- R7_FUNCTIONAL_ADJACENCY
- R8_BUDGET
- R9_CATEGORY_COVERAGE
- R10_OVERLAP
- R11_HEIGHT_CEILING
