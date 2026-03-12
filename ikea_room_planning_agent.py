"""IKEA Intelligent Room Planning Agent.

Denna modul implementerar en regelbaserad, iterativ planeringsagent som:
1) väljer möbler från ett IKEA-liknande dataset,
2) placerar dem heuristiskt i ett rum,
3) validerar mot hårda regler,
4) justerar planen vid konflikter,
5) returnerar strukturerat JSON + textsammanfattning.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_WALKWAY_MIN_M = 0.60
RECOMMENDED_PRIMARY_ROUTE_M = 0.80
DEFAULT_MAX_ITERATIONS = 5
GRID_STEP_M = 0.10
PATH_SAMPLE_STEP_M = 0.05


RULE_ROOM_BOUNDS = "R1_ROOM_BOUNDS"
RULE_WALKWAY = "R2_WALKWAY_CLEARANCE"
RULE_DESK_LEGROOM = "R3_DESK_LEGROOM"
RULE_BED_SIDE = "R4_BED_SIDE_CLEARANCE"
RULE_SOFA_FRONT = "R5_SOFA_FRONT_CLEARANCE"
RULE_DOOR_SWING = "R6_DOOR_SWING_ACCESS"
RULE_FUNCTIONAL = "R7_FUNCTIONAL_ADJACENCY"
RULE_BUDGET = "R8_BUDGET"
RULE_CATEGORY = "R9_CATEGORY_COVERAGE"
RULE_OVERLAP = "R10_OVERLAP"
RULE_HEIGHT = "R11_HEIGHT_CEILING"


@dataclass
class Rectangle:
    x_m: float
    y_m: float
    width_m: float
    length_m: float
    ref_id: str = ""
    ref_kind: str = "item"

    @property
    def x2(self) -> float:
        return self.x_m + self.width_m

    @property
    def y2(self) -> float:
        return self.y_m + self.length_m

    def center(self) -> Tuple[float, float]:
        return (self.x_m + self.width_m / 2.0, self.y_m + self.length_m / 2.0)


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    width_m: float
    length_m: float
    height_m: float
    price_SEK: float
    style_tags: List[str] = field(default_factory=list)
    convertible: bool = False
    notes: str = ""
    sku_url: str = ""

    @property
    def footprint(self) -> float:
        return self.width_m * self.length_m


@dataclass
class PlannedItem:
    product: Product
    qty: int = 1
    required: bool = True
    priority_rank: int = 10
    source_need: str = "general"


@dataclass
class Placement:
    product: Product
    x_m: float
    y_m: float
    width_m: float
    length_m: float
    orientation: str
    qty: int = 1
    zone: str = "general"
    required: bool = True
    priority_rank: int = 10
    source_need: str = "general"

    def rect(self) -> Rectangle:
        return Rectangle(
            x_m=self.x_m,
            y_m=self.y_m,
            width_m=self.width_m,
            length_m=self.length_m,
            ref_id=self.product.product_id,
            ref_kind="item",
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "product_id": self.product.product_id,
            "name": self.product.name,
            "category": self.product.category,
            "qty": self.qty,
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "width_m": round(self.width_m, 3),
            "length_m": round(self.length_m, 3),
            "height_m": round(self.product.height_m, 3),
            "orientation": self.orientation,
            "price_SEK": round(self.product.price_SEK, 2),
            "subtotal_SEK": round(self.product.price_SEK * self.qty, 2),
            "zone": self.zone,
            "required": self.required,
            "source_need": self.source_need,
        }


@dataclass
class ValidationFailure:
    rule_id: str
    rejection_reason: str
    item_ids: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rejection_reason": self.rejection_reason,
            "item_ids": self.item_ids,
        }


def _norm(s: str) -> str:
    return s.strip().lower()


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _overlap(a: Rectangle, b: Rectangle) -> bool:
    return not (a.x2 <= b.x_m or b.x2 <= a.x_m or a.y2 <= b.y_m or b.y2 <= a.y_m)


def _rect_distance(a: Rectangle, b: Rectangle) -> float:
    dx = max(a.x_m - b.x2, b.x_m - a.x2, 0.0)
    dy = max(a.y_m - b.y2, b.y_m - a.y2, 0.0)
    return math.hypot(dx, dy)


def _point_to_rect_distance(px: float, py: float, rect: Rectangle) -> float:
    dx = max(rect.x_m - px, 0.0, px - rect.x2)
    dy = max(rect.y_m - py, 0.0, py - rect.y2)
    return math.hypot(dx, dy)


def _frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    if step <= 0:
        return values
    n = int(math.floor((stop - start) / step + 1e-9))
    for i in range(max(n + 1, 0)):
        values.append(round(start + i * step, 4))
    if not values or values[-1] < stop - 1e-9:
        values.append(round(stop, 4))
    return values


class IkeaRoomPlanningAgent:
    """Regelbaserad agent för möbelplanering."""

    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        self.max_iterations = max_iterations

    def plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        room = payload.get("room", {})
        products = [self._parse_product(p) for p in payload.get("dataset", [])]
        products = [p for p in products if p is not None]

        needs = self._normalize_needs(payload.get("needs"))
        budget = _float(payload.get("budget_SEK"), 0.0)
        styles = [_norm(s) for s in payload.get("style", [])] if payload.get("style") else []
        clear_pref = payload.get("clearance_preferences", {}) or {}
        walkway_min = _float(clear_pref.get("walkway_min_m"), DEFAULT_WALKWAY_MIN_M)
        priority_order = payload.get("priority_order")
        placement_hints = payload.get("placement_hints", {}) or {}
        max_iterations = int(payload.get("max_iterations", self.max_iterations))
        if max_iterations < 1:
            max_iterations = 1

        requirement_groups = self._build_requirement_groups(needs)
        candidates = self._filter_candidates(products, room, budget, styles)
        selected, pre_rejections = self._build_initial_plan(
            needs=needs,
            requirement_groups=requirement_groups,
            candidates=candidates,
            priority_order=priority_order,
        )

        if not selected:
            failures = pre_rejections or [
                ValidationFailure(
                    rule_id=RULE_CATEGORY,
                    rejection_reason="Inga valbara produkter hittades i datasetet för behoven.",
                    item_ids=[],
                )
            ]
            return self._build_result(
                status="best_effort",
                room=room,
                needs=needs,
                budget=budget,
                styles=styles,
                iteration=0,
                placements=[],
                failures=failures,
                tradeoffs=["Datasetet saknar tillräckliga produkter för grundläggande behovstäckning."],
                alternatives=self._category_alternatives(candidates),
                walkway_min=walkway_min,
                iterations_run=0,
                adjustments_attempted=[],
            )

        best_snapshot: Optional[Dict[str, Any]] = None
        all_tradeoffs: List[str] = []
        all_failures: List[ValidationFailure] = pre_rejections[:]
        applied_adjustments: List[str] = []
        iterations_run = 0

        for iteration in range(1, max_iterations + 1):
            iterations_run = iteration
            placements, unplaced_ids = self._plan_layout(
                room=room,
                planned_items=selected,
                placement_hints=placement_hints,
            )

            validation_failures = self._validate(
                room=room,
                placements=placements,
                unplaced_ids=unplaced_ids,
                needs=needs,
                requirement_groups=requirement_groups,
                budget=budget,
                walkway_min=walkway_min,
            )

            all_failures.extend(validation_failures)

            score = self._score_solution(validation_failures, placements, budget)
            snapshot = {
                "iteration": iteration,
                "selected": copy.deepcopy(selected),
                "placements": placements,
                "failures": validation_failures,
                "tradeoffs": applied_adjustments[:],
                "score": score,
            }
            if best_snapshot is None or snapshot["score"] < best_snapshot["score"]:
                best_snapshot = snapshot

            if not validation_failures:
                return self._build_result(
                    status="success",
                    room=room,
                    needs=needs,
                    budget=budget,
                    styles=styles,
                    iteration=iteration,
                    placements=placements,
                    failures=pre_rejections,
                    tradeoffs=all_tradeoffs + applied_adjustments,
                    alternatives=self._category_alternatives(candidates),
                    walkway_min=walkway_min,
                    iterations_run=iterations_run,
                    adjustments_attempted=applied_adjustments,
                )

            changed, tradeoff_text = self._adjust_plan(
                failures=validation_failures,
                selected=selected,
                candidates=candidates,
                needs=needs,
                budget=budget,
            )
            if tradeoff_text:
                applied_adjustments.append(tradeoff_text)
            if not changed:
                all_tradeoffs.append(
                    "Automatisk justering hittade ingen förbättring i senaste iterationen; returnerar bästa möjliga plan."
                )
                break

        if best_snapshot is None:
            best_snapshot = {
                "iteration": 0,
                "placements": [],
                "failures": all_failures,
            }

        dedup_failures = self._dedup_failures(best_snapshot["failures"] + pre_rejections)
        best_tradeoffs = all_tradeoffs + best_snapshot.get("tradeoffs", [])
        if not best_tradeoffs and applied_adjustments:
            best_tradeoffs = [
                "Försökta justeringar gav inte en bättre slutplan: "
                + "; ".join(applied_adjustments[:4])
            ]
        return self._build_result(
            status="best_effort",
            room=room,
            needs=needs,
            budget=budget,
            styles=styles,
            iteration=best_snapshot["iteration"],
            placements=best_snapshot["placements"],
            failures=dedup_failures,
            tradeoffs=best_tradeoffs,
            alternatives=self._category_alternatives(candidates),
            walkway_min=walkway_min,
            iterations_run=iterations_run,
            adjustments_attempted=applied_adjustments,
        )

    def _parse_product(self, row: Dict[str, Any]) -> Optional[Product]:
        required_fields = [
            "product_id",
            "name",
            "category",
            "width_m",
            "length_m",
            "height_m",
            "price_SEK",
            "style_tags",
            "convertible",
        ]
        if not all(field in row for field in required_fields):
            return None
        return Product(
            product_id=str(row["product_id"]),
            name=str(row["name"]),
            category=_norm(str(row["category"])),
            width_m=_float(row["width_m"]),
            length_m=_float(row["length_m"]),
            height_m=_float(row["height_m"]),
            price_SEK=_float(row["price_SEK"]),
            style_tags=[_norm(x) for x in row.get("style_tags", [])],
            convertible=bool(row.get("convertible", False)),
            notes=str(row.get("notes", "")),
            sku_url=str(row.get("sku_url", "")),
        )

    def _normalize_needs(self, needs_raw: Any) -> List[str]:
        if isinstance(needs_raw, str):
            tokens = [x.strip() for x in needs_raw.replace(";", ",").split(",") if x.strip()]
            return [_norm(x) for x in tokens]
        if isinstance(needs_raw, Sequence):
            return [_norm(str(x)) for x in needs_raw if str(x).strip()]
        return []

    def _build_requirement_groups(self, needs: List[str]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for need in needs:
            if "home office" in need:
                groups.extend(
                    [
                        {"need": need, "options": ["desk"], "required": True, "priority": 1},
                        {"need": need, "options": ["chair"], "required": True, "priority": 1},
                        {"need": need, "options": ["lighting"], "required": True, "priority": 2},
                    ]
                )
            elif "guest bed" in need:
                groups.extend(
                    [
                        {
                            "need": need,
                            "options": ["bed", "sofa_convertible"],
                            "required": True,
                            "priority": 1,
                        },
                        {"need": need, "options": ["storage"], "required": True, "priority": 2},
                    ]
                )
            elif "sleep" in need:
                groups.append({"need": need, "options": ["bed"], "required": True, "priority": 1})
            elif "storage" in need:
                groups.append({"need": need, "options": ["storage"], "required": True, "priority": 1})
            elif "living" in need or "lounge" in need:
                groups.append({"need": need, "options": ["sofa"], "required": True, "priority": 1})
            else:
                # Okänd need hålls som mjuk signal (ingen hård minimikategori).
                groups.append({"need": need, "options": [], "required": False, "priority": 3})
        return groups

    def _filter_candidates(
        self,
        products: List[Product],
        room: Dict[str, Any],
        budget: float,
        styles: List[str],
    ) -> Dict[str, List[Product]]:
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))

        out: Dict[str, List[Tuple[Tuple[float, float, float, float], Product]]] = {}
        for p in products:
            fits_any_orientation = (
                (p.width_m <= room_w and p.length_m <= room_l)
                or (p.length_m <= room_w and p.width_m <= room_l)
            )
            if not fits_any_orientation:
                continue

            style_overlap = len(set(styles).intersection(set(p.style_tags))) if styles else 0
            budget_penalty = 0.0 if budget <= 0 or p.price_SEK <= budget * 0.55 else 1.0
            size_penalty = p.footprint / max(room_w * room_l, 1e-6)
            score = (-float(style_overlap), budget_penalty, p.price_SEK, size_penalty)
            out.setdefault(p.category, []).append((score, p))

            if p.category == "sofa" and p.convertible:
                alt_score = (-float(style_overlap + 1), budget_penalty, p.price_SEK, size_penalty)
                out.setdefault("sofa_convertible", []).append((alt_score, p))

        sorted_out: Dict[str, List[Product]] = {}
        for category, tuples in out.items():
            tuples.sort(key=lambda x: x[0])
            sorted_out[category] = [p for _, p in tuples]
        return sorted_out

    def _build_initial_plan(
        self,
        needs: List[str],
        requirement_groups: List[Dict[str, Any]],
        candidates: Dict[str, List[Product]],
        priority_order: Optional[Sequence[str]],
    ) -> Tuple[List[PlannedItem], List[ValidationFailure]]:
        selected: List[PlannedItem] = []
        failures: List[ValidationFailure] = []
        used_ids = set()

        priority_rank_for_need: Dict[str, int] = {}
        if priority_order:
            for idx, item in enumerate(priority_order):
                priority_rank_for_need[_norm(str(item))] = idx

        for req in requirement_groups:
            options = req.get("options", [])
            if not options:
                continue
            chosen: Optional[Product] = None
            chosen_option = ""
            for option in options:
                pool = candidates.get(option, [])
                for p in pool:
                    if p.product_id in used_ids:
                        continue
                    chosen = p
                    chosen_option = option
                    break
                if chosen:
                    break

            if chosen is None:
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_CATEGORY,
                        rejection_reason=(
                            f"Saknar kandidat i kategori/alternativ {options} för behov '{req['need']}'."
                        ),
                        item_ids=[],
                    )
                )
                continue

            priority = req.get("priority", 2)
            need_norm = _norm(req.get("need", "general"))
            for signal, rank in priority_rank_for_need.items():
                if signal in need_norm:
                    priority = min(priority, rank)
            selected.append(
                PlannedItem(
                    product=chosen,
                    qty=1,
                    required=bool(req.get("required", True)),
                    priority_rank=int(priority),
                    source_need=req.get("need", chosen_option),
                )
            )
            used_ids.add(chosen.product_id)

        # Mjuk komplettering: om behov inkluderar office, lägg till extra förvaring om möjligt.
        if any("office" in n for n in needs) and "storage" in candidates:
            for p in candidates["storage"]:
                if p.product_id not in used_ids:
                    selected.append(
                        PlannedItem(
                            product=p,
                            qty=1,
                            required=False,
                            priority_rank=4,
                            source_need="optional office storage",
                        )
                    )
                    used_ids.add(p.product_id)
                    break

        return selected, failures

    def _infer_door_rect(self, room: Dict[str, Any]) -> Optional[Rectangle]:
        door = room.get("door")
        if not door:
            return None
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))
        x = _float(door.get("x_m"))
        y = _float(door.get("y_m"))
        w = _float(door.get("width_m"))
        if w <= 0:
            return None

        d_bottom = abs(y - 0.0)
        d_top = abs(y - room_l)
        d_left = abs(x - 0.0)
        d_right = abs(x - room_w)
        wall = min(
            [("bottom", d_bottom), ("top", d_top), ("left", d_left), ("right", d_right)],
            key=lambda t: t[1],
        )[0]

        if wall == "bottom":
            return Rectangle(x_m=x, y_m=0.0, width_m=w, length_m=w, ref_id="door_swing", ref_kind="door")
        if wall == "top":
            return Rectangle(
                x_m=x,
                y_m=max(room_l - w, 0.0),
                width_m=w,
                length_m=w,
                ref_id="door_swing",
                ref_kind="door",
            )
        if wall == "left":
            return Rectangle(x_m=0.0, y_m=y, width_m=w, length_m=w, ref_id="door_swing", ref_kind="door")
        return Rectangle(
            x_m=max(room_w - w, 0.0),
            y_m=y,
            width_m=w,
            length_m=w,
            ref_id="door_swing",
            ref_kind="door",
        )

    def _fixed_feature_rects(self, room: Dict[str, Any]) -> List[Rectangle]:
        rects: List[Rectangle] = []
        for i, feat in enumerate(room.get("fixed_features", []) or []):
            rects.append(
                Rectangle(
                    x_m=_float(feat.get("x_m")),
                    y_m=_float(feat.get("y_m")),
                    width_m=_float(feat.get("width_m")),
                    length_m=_float(feat.get("length_m")),
                    ref_id=f"fixed_{i}",
                    ref_kind="fixed",
                )
            )
        return rects

    def _orientations(self, product: Product) -> List[Tuple[float, float, str]]:
        if abs(product.width_m - product.length_m) < 1e-6:
            return [(product.width_m, product.length_m, "default")]
        return [
            (product.width_m, product.length_m, "default"),
            (product.length_m, product.width_m, "rotated"),
        ]

    def _plan_layout(
        self,
        room: Dict[str, Any],
        planned_items: List[PlannedItem],
        placement_hints: Dict[str, Any],
    ) -> Tuple[List[Placement], List[str]]:
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))
        fixed_rects = self._fixed_feature_rects(room)
        door_rect = self._infer_door_rect(room)
        if door_rect:
            fixed_rects.append(door_rect)

        ordered = sorted(
            planned_items,
            key=lambda it: (
                it.priority_rank,
                -it.product.footprint,
                it.product.price_SEK,
            ),
        )

        placements: List[Placement] = []
        unplaced: List[str] = []
        desk_placement: Optional[Placement] = None

        for item in ordered:
            product = item.product
            best: Optional[Tuple[float, Placement]] = None

            for ow, ol, orientation in self._orientations(product):
                if ow > room_w or ol > room_l:
                    continue

                xs = _frange(0.0, max(room_w - ow, 0.0), GRID_STEP_M)
                ys = _frange(0.0, max(room_l - ol, 0.0), GRID_STEP_M)
                for y in ys:
                    for x in xs:
                        rect = Rectangle(x_m=x, y_m=y, width_m=ow, length_m=ol, ref_id=product.product_id)
                        blocked = any(_overlap(rect, r) for r in fixed_rects)
                        if blocked:
                            continue
                        if any(_overlap(rect, p.rect()) for p in placements):
                            continue

                        placement = Placement(
                            product=product,
                            x_m=x,
                            y_m=y,
                            width_m=ow,
                            length_m=ol,
                            orientation=orientation,
                            qty=item.qty,
                            zone=self._zone_for_item(item),
                            required=item.required,
                            priority_rank=item.priority_rank,
                            source_need=item.source_need,
                        )
                        score = self._placement_score(
                            room=room,
                            placement=placement,
                            placements=placements,
                            desk_placement=desk_placement,
                            placement_hints=placement_hints,
                        )
                        if best is None or score < best[0]:
                            best = (score, placement)

            if best is None:
                unplaced.append(product.product_id)
                continue

            placements.append(best[1])
            if product.category == "desk":
                desk_placement = best[1]

        return placements, unplaced

    def _zone_for_item(self, item: PlannedItem) -> str:
        need = _norm(item.source_need)
        if "office" in need:
            return "work"
        if "guest bed" in need or "sleep" in need:
            return "sleep"
        if "storage" in need:
            return "storage"
        return "general"

    def _placement_score(
        self,
        room: Dict[str, Any],
        placement: Placement,
        placements: List[Placement],
        desk_placement: Optional[Placement],
        placement_hints: Dict[str, Any],
    ) -> float:
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))
        rect = placement.rect()
        cx, cy = rect.center()

        # Grundkostnad: håll viss marginal till väggar (men tillåt väggnära placering för stora möbler).
        min_wall_dist = min(cx, cy, room_w - cx, room_l - cy)
        score = max(0.0, 0.6 - min_wall_dist)

        if placement.product.category in {"bed", "sofa", "storage"}:
            # Dessa placeras gärna mot vägg.
            wall_proximity = min(rect.x_m, rect.y_m, room_w - rect.x2, room_l - rect.y2)
            score += wall_proximity

        if placement.product.category == "chair" and desk_placement:
            score += _rect_distance(rect, desk_placement.rect()) * 2.0

        hint = str(placement_hints.get("prefer_desk_under_window", "")).lower()
        if placement.product.category == "desk" and hint in {"1", "true", "yes"}:
            windows = room.get("windows", []) or []
            if windows:
                wx_centers = [
                    (_float(w.get("x_m")) + _float(w.get("width_m")) / 2.0, _float(w.get("y_m")))
                    for w in windows
                ]
                nearest = min(math.hypot(cx - wx, cy - wy) for wx, wy in wx_centers)
                score += nearest

        # Lätt bonus för att skapa separata zoner (work/sleep) om möjligt.
        if placement.zone == "work":
            for other in placements:
                if other.zone == "sleep":
                    score += max(0.0, 1.2 - _rect_distance(rect, other.rect())) * 2.0
        if placement.zone == "sleep":
            for other in placements:
                if other.zone == "work":
                    score += max(0.0, 1.2 - _rect_distance(rect, other.rect())) * 2.0

        return score

    def _validate(
        self,
        room: Dict[str, Any],
        placements: List[Placement],
        unplaced_ids: List[str],
        needs: List[str],
        requirement_groups: List[Dict[str, Any]],
        budget: float,
        walkway_min: float,
    ) -> List[ValidationFailure]:
        failures: List[ValidationFailure] = []
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))
        fixed_rects = self._fixed_feature_rects(room)
        door_rect = self._infer_door_rect(room)

        # R1: bounds + unplaced signal
        for p in placements:
            if p.x_m < 0 or p.y_m < 0 or p.x_m + p.width_m > room_w + 1e-9 or p.y_m + p.length_m > room_l + 1e-9:
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_ROOM_BOUNDS,
                        rejection_reason=f"{p.product.product_id} ligger delvis utanför rummets gränser.",
                        item_ids=[p.product.product_id],
                    )
                )
        for product_id in unplaced_ids:
            failures.append(
                ValidationFailure(
                    rule_id=RULE_ROOM_BOUNDS,
                    rejection_reason=f"{product_id} kunde inte placeras inom rumsgränser/ledigt utrymme.",
                    item_ids=[product_id],
                )
            )

        # R10: overlap mellan items och fixed features
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                if _overlap(placements[i].rect(), placements[j].rect()):
                    failures.append(
                        ValidationFailure(
                            rule_id=RULE_OVERLAP,
                            rejection_reason=(
                                f"{placements[i].product.product_id} överlappar {placements[j].product.product_id}."
                            ),
                            item_ids=[placements[i].product.product_id, placements[j].product.product_id],
                        )
                    )
        for p in placements:
            for f in fixed_rects:
                if _overlap(p.rect(), f):
                    failures.append(
                        ValidationFailure(
                            rule_id=RULE_OVERLAP,
                            rejection_reason=f"{p.product.product_id} överlappar fast feature ({f.ref_id}).",
                            item_ids=[p.product.product_id],
                        )
                    )

        # R6: door swing
        if door_rect:
            for p in placements:
                if _overlap(p.rect(), door_rect):
                    failures.append(
                        ValidationFailure(
                            rule_id=RULE_DOOR_SWING,
                            rejection_reason=f"{p.product.product_id} blockerar dörrens svängyta.",
                            item_ids=[p.product.product_id],
                        )
                    )

        # R2: walkway/circulation
        walkway_fail = self._check_walkway_clearance(room, placements, fixed_rects, walkway_min)
        if walkway_fail:
            failures.append(walkway_fail)

        # R3 desk legroom
        for p in placements:
            if p.product.category != "desk":
                continue
            front_rect = Rectangle(
                x_m=p.x_m,
                y_m=p.y_m + p.length_m,
                width_m=p.width_m,
                length_m=0.60,
                ref_id=f"{p.product.product_id}_desk_front",
            )
            if front_rect.y2 > room_l + 1e-9:
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_DESK_LEGROOM,
                        rejection_reason=f"{p.product.product_id} saknar 0.60m front clearance.",
                        item_ids=[p.product.product_id],
                    )
                )
                continue
            collides = any(_overlap(front_rect, q.rect()) for q in placements if q.product.product_id != p.product.product_id)
            collides = collides or any(_overlap(front_rect, f) for f in fixed_rects)
            if collides:
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_DESK_LEGROOM,
                        rejection_reason=f"{p.product.product_id} har blockerad benplats framför skrivbordet.",
                        item_ids=[p.product.product_id],
                    )
                )

        # R4 bed side clearance
        for p in placements:
            if p.product.category != "bed":
                continue
            left = Rectangle(x_m=p.x_m - 0.50, y_m=p.y_m, width_m=0.50, length_m=p.length_m)
            right = Rectangle(x_m=p.x_m + p.width_m, y_m=p.y_m, width_m=0.50, length_m=p.length_m)
            left_ok = self._clear_strip_in_room(left, room_w, room_l, placements, fixed_rects, p.product.product_id)
            right_ok = self._clear_strip_in_room(right, room_w, room_l, placements, fixed_rects, p.product.product_id)
            if not (left_ok or right_ok):
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_BED_SIDE,
                        rejection_reason=f"{p.product.product_id} saknar minst 0.5m sidofrigång på minst en sida.",
                        item_ids=[p.product.product_id],
                    )
                )

        # R5 sofa front
        for p in placements:
            if p.product.category != "sofa":
                continue
            front = Rectangle(x_m=p.x_m, y_m=p.y_m + p.length_m, width_m=p.width_m, length_m=0.60)
            if front.y2 > room_l + 1e-9:
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_SOFA_FRONT,
                        rejection_reason=f"{p.product.product_id} saknar 0.6m frontfriyta.",
                        item_ids=[p.product.product_id],
                    )
                )
                continue
            if any(_overlap(front, q.rect()) for q in placements if q.product.product_id != p.product.product_id):
                failures.append(
                    ValidationFailure(
                        rule_id=RULE_SOFA_FRONT,
                        rejection_reason=f"{p.product.product_id} har blockerad frontfriyta.",
                        item_ids=[p.product.product_id],
                    )
                )

        # R9 category coverage
        coverage_failure = self._validate_category_coverage(placements, requirement_groups)
        if coverage_failure:
            failures.append(coverage_failure)

        # R7 functional adjacency
        functional_failure = self._validate_functional_adjacency(room, placements, needs)
        if functional_failure:
            failures.append(functional_failure)

        # R8 budget
        total_cost = sum(p.product.price_SEK * p.qty for p in placements)
        if budget > 0 and total_cost > budget + 1e-9:
            failures.append(
                ValidationFailure(
                    rule_id=RULE_BUDGET,
                    rejection_reason=f"Totalkostnad {total_cost:.0f} SEK överstiger budget {budget:.0f} SEK.",
                    item_ids=[p.product.product_id for p in placements],
                )
            )

        # R11 ceiling height
        ceiling = room.get("floor_to_ceiling_m")
        if ceiling is not None:
            c = _float(ceiling)
            for p in placements:
                if p.product.height_m > c + 1e-9:
                    failures.append(
                        ValidationFailure(
                            rule_id=RULE_HEIGHT,
                            rejection_reason=f"{p.product.product_id} är för hög ({p.product.height_m}m > {c}m).",
                            item_ids=[p.product.product_id],
                        )
                    )

        return self._dedup_failures(failures)

    def _check_walkway_clearance(
        self,
        room: Dict[str, Any],
        placements: List[Placement],
        fixed_rects: List[Rectangle],
        walkway_min: float,
    ) -> Optional[ValidationFailure]:
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))
        door = room.get("door")
        if door:
            start = (
                _float(door.get("x_m")) + _float(door.get("width_m")) / 2.0,
                _float(door.get("y_m")) + 0.05,
            )
        else:
            start = (room_w / 2.0, 0.05)

        anchors: List[Tuple[float, float, str]] = []
        for p in placements:
            if p.product.category in {"desk", "bed", "sofa"}:
                cx, cy = p.rect().center()
                anchors.append((cx, cy, p.product.product_id))

        if not anchors:
            anchors = [(room_w / 2.0, room_l / 2.0, "room_center")]

        obstacle_rects = [p.rect() for p in placements] + fixed_rects
        radius = max(walkway_min / 2.0, DEFAULT_WALKWAY_MIN_M / 2.0)

        for tx, ty, target_id in anchors:
            dx = tx - start[0]
            dy = ty - start[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue
            samples = max(2, int(dist / PATH_SAMPLE_STEP_M))
            # Sista 10% tillåts vara nära målobjektet.
            check_until = int(samples * 0.9)
            for i in range(1, max(check_until, 1)):
                t = i / float(samples)
                px = start[0] + dx * t
                py = start[1] + dy * t
                if px < radius or py < radius or px > room_w - radius or py > room_l - radius:
                    return ValidationFailure(
                        rule_id=RULE_WALKWAY,
                        rejection_reason=(
                            f"Otillräcklig gångyta (<{walkway_min:.2f}m) i huvudstråk mot {target_id}."
                        ),
                        item_ids=[target_id] if target_id != "room_center" else [],
                    )
                near_obstacle = any(_point_to_rect_distance(px, py, r) < radius - 1e-9 for r in obstacle_rects)
                if near_obstacle:
                    return ValidationFailure(
                        rule_id=RULE_WALKWAY,
                        rejection_reason=(
                            f"Gångstråk mot {target_id} underskrider minsta fria bredd {walkway_min:.2f}m."
                        ),
                        item_ids=[target_id] if target_id != "room_center" else [],
                    )
        return None

    def _clear_strip_in_room(
        self,
        strip: Rectangle,
        room_w: float,
        room_l: float,
        placements: List[Placement],
        fixed_rects: List[Rectangle],
        own_product_id: str,
    ) -> bool:
        if strip.width_m <= 0 or strip.length_m <= 0:
            return False
        if strip.x_m < 0 or strip.y_m < 0 or strip.x2 > room_w + 1e-9 or strip.y2 > room_l + 1e-9:
            return False
        if any(_overlap(strip, p.rect()) for p in placements if p.product.product_id != own_product_id):
            return False
        if any(_overlap(strip, f) for f in fixed_rects):
            return False
        return True

    def _validate_category_coverage(
        self,
        placements: List[Placement],
        requirement_groups: List[Dict[str, Any]],
    ) -> Optional[ValidationFailure]:
        selected_categories = [p.product.category for p in placements]
        has_sofa_convertible = any(p.product.category == "sofa" and p.product.convertible for p in placements)

        for req in requirement_groups:
            if not req.get("required", True):
                continue
            options = req.get("options", [])
            if not options:
                continue
            satisfied = False
            for option in options:
                if option == "sofa_convertible":
                    satisfied = satisfied or has_sofa_convertible
                else:
                    satisfied = satisfied or (option in selected_categories)
            if not satisfied:
                return ValidationFailure(
                    rule_id=RULE_CATEGORY,
                    rejection_reason=(
                        f"Behov '{req.get('need')}' saknar minimikategori. Kräver ett av: {options}."
                    ),
                    item_ids=[],
                )
        return None

    def _validate_functional_adjacency(
        self,
        room: Dict[str, Any],
        placements: List[Placement],
        needs: List[str],
    ) -> Optional[ValidationFailure]:
        need_home_office = any("home office" in n for n in needs)
        need_guest_bed = any("guest bed" in n for n in needs)
        if not (need_home_office and need_guest_bed):
            return None

        desks = [p for p in placements if p.product.category == "desk"]
        chairs = [p for p in placements if p.product.category == "chair"]
        beds = [p for p in placements if p.product.category == "bed"]
        sofas = [p for p in placements if p.product.category == "sofa"]
        convertible = [p for p in sofas if p.product.convertible]

        if not desks or not chairs:
            return ValidationFailure(
                rule_id=RULE_FUNCTIONAL,
                rejection_reason="Work zone saknar desk+chair för home office.",
                item_ids=[],
            )

        # Desk + chair närhet
        nearest_chair_dist = min(_rect_distance(desks[0].rect(), c.rect()) for c in chairs)
        if nearest_chair_dist > 1.0:
            return ValidationFailure(
                rule_id=RULE_FUNCTIONAL,
                rejection_reason="Chair är inte funktionellt nära desk i work zone.",
                item_ids=[desks[0].product.product_id],
            )

        # Desk nära vägg (proximity to power antas via vägguttag)
        room_w = _float(room.get("width_m"))
        room_l = _float(room.get("length_m"))
        d = desks[0]
        wall_dist = min(d.x_m, d.y_m, room_w - (d.x_m + d.width_m), room_l - (d.y_m + d.length_m))
        if wall_dist > 0.6:
            return ValidationFailure(
                rule_id=RULE_FUNCTIONAL,
                rejection_reason="Desk är för långt från vägg (>0.6m), osäker el-anslutning.",
                item_ids=[d.product.product_id],
            )

        if not beds and not convertible:
            return ValidationFailure(
                rule_id=RULE_FUNCTIONAL,
                rejection_reason="Guest bed saknas (kräver bed eller convertible sofa).",
                item_ids=[],
            )

        # Distinkta zoner om inte convertible används.
        if not convertible and beds:
            bed = beds[0]
            dist = _rect_distance(desks[0].rect(), bed.rect())
            if dist < 1.2:
                return ValidationFailure(
                    rule_id=RULE_FUNCTIONAL,
                    rejection_reason=(
                        "Home office och guest bed är inte tydligt zonindelade (minst 1.2m rekommenderas)."
                    ),
                    item_ids=[desks[0].product.product_id, bed.product.product_id],
                )

        return None

    def _score_solution(self, failures: List[ValidationFailure], placements: List[Placement], budget: float) -> float:
        if not failures:
            return 0.0
        severity = {
            RULE_OVERLAP: 5.0,
            RULE_ROOM_BOUNDS: 5.0,
            RULE_DOOR_SWING: 4.0,
            RULE_CATEGORY: 4.0,
            RULE_FUNCTIONAL: 4.0,
            RULE_WALKWAY: 3.0,
            RULE_BUDGET: 3.0,
            RULE_DESK_LEGROOM: 2.0,
            RULE_BED_SIDE: 2.0,
            RULE_SOFA_FRONT: 2.0,
            RULE_HEIGHT: 2.0,
        }
        score = 0.0
        for f in failures:
            score += severity.get(f.rule_id, 1.0)
        if budget > 0:
            total = sum(p.product.price_SEK * p.qty for p in placements)
            score += max(0.0, (total - budget) / max(budget, 1.0))
        return score

    def _adjust_plan(
        self,
        failures: List[ValidationFailure],
        selected: List[PlannedItem],
        candidates: Dict[str, List[Product]],
        needs: List[str],
        budget: float,
    ) -> Tuple[bool, str]:
        if not failures:
            return False, ""

        first = failures[0]
        by_product = {item.product.product_id: item for item in selected}

        # 1) Budget: byt till billigare / minska
        if first.rule_id == RULE_BUDGET:
            changed = self._replace_with_cheaper(selected, candidates)
            if changed:
                return True, "Bytte minst en produkt till billigare alternativ för att hålla budget."
            removed = self._remove_lowest_priority(selected)
            if removed:
                return True, f"Tog bort lågprioriterad produkt ({removed}) för budgetbalans."
            return False, ""

        # 2) Clearance/bounds/overlap/door: ersätt problemprodukt med mindre alternativ.
        if first.rule_id in {RULE_OVERLAP, RULE_ROOM_BOUNDS, RULE_DOOR_SWING, RULE_WALKWAY, RULE_DESK_LEGROOM, RULE_BED_SIDE, RULE_SOFA_FRONT}:
            for pid in first.item_ids:
                item = by_product.get(pid)
                if not item:
                    continue
                smaller = self._pick_smaller_alternative(item.product, candidates)
                if smaller:
                    item.product = smaller
                    return True, f"Ersatte {pid} med mindre alternativ för att lösa utrymmeskonflikt."
            removed = self._remove_lowest_priority(selected)
            if removed:
                return True, f"Plockade bort lågprioriterad produkt ({removed}) för att frigöra yta."
            return False, ""

        # 3) Funktionskonflikt: försök convertible
        if first.rule_id in {RULE_FUNCTIONAL, RULE_CATEGORY} and any("guest bed" in n for n in needs):
            convertible_pool = candidates.get("sofa_convertible", [])
            if convertible_pool:
                # Ersätt vanlig bed eller sofa med convertible sofa.
                bed_idx = next((i for i, it in enumerate(selected) if it.product.category == "bed"), None)
                sofa_idx = next((i for i, it in enumerate(selected) if it.product.category == "sofa"), None)
                replacement = convertible_pool[0]
                if bed_idx is not None:
                    selected[bed_idx].product = replacement
                    selected[bed_idx].source_need = "guest bed (convertible)"
                    return True, "Bytte guest bed till convertible sofa för bättre behovskombination."
                if sofa_idx is not None:
                    selected[sofa_idx].product = replacement
                    selected[sofa_idx].source_need = "guest bed (convertible)"
                    return True, "Bytte sofa till convertible variant för att möta guest bed-krav."
                selected.append(
                    PlannedItem(
                        product=replacement,
                        qty=1,
                        required=True,
                        priority_rank=2,
                        source_need="guest bed (convertible)",
                    )
                )
                return True, "La till convertible sofa för att möta gästbäddsbehov."

        # 4) Fallback
        changed = self._replace_with_cheaper(selected, candidates)
        if changed:
            return True, "Fallback: bytte till billigare alternativ."
        removed = self._remove_lowest_priority(selected)
        if removed:
            return True, f"Fallback: tog bort lågprioriterad produkt ({removed})."
        return False, ""

    def _pick_smaller_alternative(
        self,
        product: Product,
        candidates: Dict[str, List[Product]],
    ) -> Optional[Product]:
        pool = candidates.get(product.category, [])
        for alt in pool:
            if alt.product_id == product.product_id:
                continue
            if alt.footprint < product.footprint - 1e-9:
                return alt
        return None

    def _replace_with_cheaper(self, selected: List[PlannedItem], candidates: Dict[str, List[Product]]) -> bool:
        # Försök byta dyraste produkt först.
        sorted_items = sorted(selected, key=lambda x: x.product.price_SEK, reverse=True)
        for item in sorted_items:
            pool = candidates.get(item.product.category, [])
            for alt in pool:
                if alt.product_id == item.product.product_id:
                    continue
                if alt.price_SEK < item.product.price_SEK - 1e-9:
                    item.product = alt
                    return True
        return False

    def _remove_lowest_priority(self, selected: List[PlannedItem]) -> Optional[str]:
        if not selected:
            return None
        optional = [it for it in selected if not it.required]
        pool = optional if optional else selected
        victim = sorted(pool, key=lambda x: (x.priority_rank, x.product.price_SEK), reverse=True)[0]
        selected.remove(victim)
        return victim.product.product_id

    def _dedup_failures(self, failures: List[ValidationFailure]) -> List[ValidationFailure]:
        seen = set()
        out: List[ValidationFailure] = []
        for f in failures:
            key = (f.rule_id, f.rejection_reason, tuple(sorted(f.item_ids)))
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def _category_alternatives(self, candidates: Dict[str, List[Product]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for category, items in sorted(candidates.items()):
            short = items[:3]
            out.append(
                {
                    "category": category,
                    "alternatives": [
                        {
                            "product_id": p.product_id,
                            "name": p.name,
                            "price_SEK": p.price_SEK,
                            "width_m": p.width_m,
                            "length_m": p.length_m,
                            "convertible": p.convertible,
                        }
                        for p in short
                    ],
                }
            )
        return out

    def _build_result(
        self,
        status: str,
        room: Dict[str, Any],
        needs: List[str],
        budget: float,
        styles: List[str],
        iteration: int,
        placements: List[Placement],
        failures: List[ValidationFailure],
        tradeoffs: List[str],
        alternatives: List[Dict[str, Any]],
        walkway_min: float,
        iterations_run: Optional[int] = None,
        adjustments_attempted: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        total_cost = sum(p.product.price_SEK * p.qty for p in placements)
        budget_ok = (budget <= 0) or (total_cost <= budget + 1e-9)

        remaining_risks: List[str] = []
        if walkway_min < RECOMMENDED_PRIMARY_ROUTE_M:
            remaining_risks.append(
                f"Gångyta validerad mot minimum {DEFAULT_WALKWAY_MIN_M:.1f}m, men {RECOMMENDED_PRIMARY_ROUTE_M:.1f}m rekommenderas för primärstråk."
            )
        if status != "success":
            remaining_risks.append("Planen är best effort och kan kräva manuell finjustering vid faktisk installation.")

        summary = self._human_summary(
            status=status,
            needs=needs,
            budget=budget,
            total_cost=total_cost,
            placement_count=len(placements),
            failures=failures,
            tradeoffs=tradeoffs,
        )

        return {
            "status": status,
            "iterations_used": iterations_run if iterations_run is not None else iteration,
            "best_iteration": iteration,
            "input_echo": {
                "room": room,
                "needs": needs,
                "budget_SEK": budget,
                "style": styles,
            },
            "proposal": {
                "items": [p.to_json() for p in placements],
                "total_cost_SEK": round(total_cost, 2),
                "budget_ok": budget_ok,
            },
            "validation": {
                "passes_all_rules": len(failures) == 0,
                "failures": [f.to_json() for f in failures],
            },
            "tradeoffs": tradeoffs,
            "adjustments_attempted": adjustments_attempted or [],
            "remaining_risks": remaining_risks,
            "alternatives": alternatives,
            "summary_text": summary,
        }

    def _human_summary(
        self,
        status: str,
        needs: List[str],
        budget: float,
        total_cost: float,
        placement_count: int,
        failures: List[ValidationFailure],
        tradeoffs: List[str],
    ) -> str:
        if status == "success":
            return (
                f"Planen uppfyller behoven ({', '.join(needs)}) med {placement_count} produkter. "
                f"Budgetutfall: {total_cost:.0f} SEK av {budget:.0f} SEK. "
                f"Validering passerade utan regelbrott. "
                + (
                    f"Viktiga trade-offs: {'; '.join(tradeoffs[:3])}."
                    if tradeoffs
                    else "Inga större trade-offs krävdes."
                )
            )
        short_fails = "; ".join(f"{f.rule_id}: {f.rejection_reason}" for f in failures[:3])
        return (
            f"Best-effort-plan framtagen med {placement_count} produkter och kostnad {total_cost:.0f} SEK "
            f"(budget {budget:.0f} SEK). "
            f"Alla krav kunde inte uppfyllas: {short_fails}. "
            + (
                f"Gjorda trade-offs: {'; '.join(tradeoffs[:4])}."
                if tradeoffs
                else "Ytterligare manuella avvägningar behövs."
            )
        )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="IKEA Intelligent Room Planning Agent")
    parser.add_argument("input_json", help="Path till input-json")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    agent = IkeaRoomPlanningAgent(max_iterations=args.max_iterations)
    result = agent.plan(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli()
