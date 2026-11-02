"""Phase 19.2 — deterministic Mineflayer *player* trajectories.

The model is trained on complete player turns, not on a bespoke command DSL:

    Request: <what the player wants>
    Thought: <why this is the next move>
    Action: <Mineflayer JavaScript>
    Observation: <what the mock world reported>
    Answer: <what the player says afterwards>

Every action is JavaScript for the real Mineflayer API (pathfinder, dig, craft,
attack, placeBlock, equip, toss, consume, chat, openContainer, farming). Each
trajectory also carries the ``mock`` state and ``expect`` assertion used by
``notebook/test_mineflayer.py``; nothing here executes Minecraft and nothing
here is a strategy the model has actually learned.

Coverage: 8 skills, >500 trajectories, deterministic for a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SKILLS = ('navigation', 'mining', 'crafting', 'combat', 'building',
          'inventory', 'chat', 'farming')

# Item ids only matter for ``bot.registry.itemsByName[...].id`` lookups. Real
# Minecraft ids are protocol/version specific; the mock world needs stable ids,
# not authentic ones. Trajectories never claim a specific server version.
ITEM_NAMES = (
    'air', 'stone', 'cobblestone', 'dirt', 'grass_block', 'sand', 'gravel',
    'oak_log', 'oak_planks', 'stick', 'crafting_table', 'furnace', 'chest',
    'torch', 'ladder', 'bucket', 'water_bucket', 'bread', 'wheat', 'wheat_seeds',
    'carrot', 'carrots', 'potato', 'potatoes', 'beetroot', 'beetroot_seeds',
    'sugar_cane', 'bone_meal', 'diamond', 'diamond_ore', 'iron_ore', 'gold_ore',
    'coal_ore', 'redstone_ore', 'emerald_ore', 'ancient_debris', 'obsidian',
    'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe',
    'wooden_sword', 'stone_sword', 'iron_sword', 'diamond_sword', 'bow', 'arrow',
    'iron_helmet', 'iron_chestplate', 'iron_leggings', 'iron_boots',
    'stone_axe', 'iron_axe', 'wooden_axe', 'wooden_shovel', 'stone_shovel', 'iron_shovel',
    'shears', 'bowl', 'cake', 'iron_ingot', 'coal', 'redstone', 'wooden_hoe', 'stone_hoe', 'iron_hoe', 'fishing_rod', 'shield', 'apple',
    'cooked_beef', 'golden_apple', 'ender_pearl', 'bucket_of_water',
)
ITEM_IDS = {name: i + 1 for i, name in enumerate(ITEM_NAMES)}

# Blocks the mock world can materialise. Everything not listed is air.
BLOCK_PROPS = {'wheat': {'age': 7}, 'carrots': {'age': 7}, 'potatoes': {'age': 7},
               'beetroot': {'age': 3}, 'farmland': {'moisture': 7}}


@dataclass(frozen=True)
class Trajectory:
    skill: str
    scenario: str
    thought: str
    action: str
    observation: str
    answer: str
    mock: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)

    def render(self) -> str:
        """Render one trainable UTF-8 byte record (no tokenizer involved)."""
        return (f'Request: {self.scenario}\n'
                f'Thought: {self.thought}\n'
                f'Action: {self.action}\n'
                f'Observation: {self.observation}\n'
                f'Answer: {self.answer}')

    def to_dict(self) -> dict:
        return {'skill': self.skill, 'scenario': self.scenario, 'thought': self.thought,
                'action': self.action, 'observation': self.observation,
                'answer': self.answer, 'mock': self.mock, 'expect': self.expect}


def mock(position=(0, 64, 0), world=None, inventory=(), entities=(), **extra) -> dict:
    """A minimal, explicit mock world. Registry ids are shared, never copied."""
    data = {'position': list(position), 'world': dict(world or {}),
            'inventory': [{'name': n, 'count': 1} for n in inventory],
            'entities': [dict(e) for e in entities], 'registry': ITEM_IDS}
    data.update(extra)
    return data


def _key(x, y, z) -> str:
    return f'{int(x)},{int(y)},{int(z)}'


def _entity(name, x, y, z, kind='mob', health=20) -> dict:
    return {'id': ITEM_IDS.get(name, 1), 'name': name, 'type': kind,
            'position': [x, y, z], 'health': health}


def _q(text: str) -> str:
    """Single-quoted JavaScript string literal."""
    return "'" + text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', ' ') + "'"


def _equip_hand(item: str) -> str:
    return (f"await bot.equip(bot.inventory.findInventoryItem("
            f"bot.registry.itemsByName['{item}'].id, null), 'hand');")


def _find_entity(mob: str, distance: int = 16) -> str:
    return (f"const target = bot.nearestEntity(e => e.type === 'mob' && e.name === "
            f"'{mob}' && e.position.distanceTo(bot.entity.position) < {distance});")


# --------------------------------------------------------------------------
# Skill generators
# --------------------------------------------------------------------------
def _navigation(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    targets = [(100, 64, 200), (0, 64, 0), (-37, 70, 12), (512, 63, -256),
               (230, 80, 88), (-120, 72, -340), (64, 64, 64), (-8, 11, 8),
               (2, 120, -15), (900, 45, 1200)]
    for x, y, z in targets:
        for radius in (1, 2, 3):
            out.append(Trajectory(
                'navigation', f'Walk to ({x}, {y}, {z}) and stop within {radius} block(s).',
                f'I need to reach ({x}, {y}, {z}); pathfinder is loaded, so plan around terrain instead of walking into it.',
                f'await bot.pathfinder.goto(new GoalNear({x}, {y}, {z}, {radius}));',
                f'Path complete; position is ({x}, {y}, {z}) within {radius} block(s).',
                f'I moved to ({x}, {y}, {z}).',
                mock=mock(), expect={'type': 'goto', 'target': [x, y, z], 'within': radius}))
        out.append(Trajectory(
            'navigation', f'Stand exactly on block ({x}, {y}, {z}).',
            'The request names a block coordinate, so GoalBlock is stricter than GoalNear.',
            f'await bot.pathfinder.goto(new GoalBlock({x}, {y}, {z}));',
            f'Reached block ({x}, {y}, {z}).', f'I am standing on ({x}, {y}, {z}).',
            mock=mock(), expect={'type': 'goto', 'target': [x, y, z]}))
        out.append(Trajectory(
            'navigation', f'Trek to x={x}, z={z}, ignoring the exact height.',
            'The exact y is unknown (caves or hills), so GoalXZ lets the pathfinder choose the height.',
            f'await bot.pathfinder.goto(new GoalXZ({x}, {z}));',
            f'Arrived above x={x}, z={z}.', f'I reached x={x}, z={z}.',
            mock=mock(), expect={'type': 'goto', 'target': [x, None, z], 'xz': True}))
    for name in ('Steve', 'Alex', 'Gabriel', 'Builder_2'):
        for radius in (2, 3):
            out.append(Trajectory(
                'navigation', f'Follow {name} and keep within {radius} blocks.',
                f'{name} is a teammate; following keeps the group together without standing inside them.',
                f"const player = bot.players['{name}'].entity;\n"
                f"if (!player) throw new Error('{name} is not visible');\n"
                f'await bot.pathfinder.goto(new GoalFollow(player, {radius}));',
                f'Following {name} at range {radius}.', f'I am following {name}.',
                mock=mock(entities=[_entity(name, 10, 64, 10, 'player')]),
                expect={'type': 'goto', 'follow': name, 'within': radius}))
    for x, y, z in targets[:4]:
        out.append(Trajectory(
            'navigation', f'Start moving towards ({x}, {y}, {z}) without blocking the chat loop.',
            'setGoal is non-blocking, so other events can still be handled while walking.',
            f'bot.pathfinder.setGoal(new GoalNear({x}, {y}, {z}, 2));',
            f'Goal set towards ({x}, {y}, {z}).', 'I am on my way.',
            mock=mock(), expect={'type': 'setGoal', 'target': [x, y, z]}))
        out.append(Trajectory(
            'navigation', 'Stop walking immediately; a player is in danger ahead.',
            'Stopping is a player decision too: continue only after the hazard is clear.',
            'bot.pathfinder.stop();\nbot.clearControlStates();',
            'Movement stopped.', 'I stopped moving.',
            mock=mock(), expect={'type': 'stop'}))
    for base in ((24, 64, 24), (-48, 70, 8), (128, 63, -64), (12, 64, 300)):
        for offset in (16, 32):
            bx, by, bz = base
            out.append(Trajectory(
                'navigation', f'Return to base at ({bx}, {by}, {bz}) from {offset} blocks away.',
                'The base coordinates are known; walking back before night beats exploring further.',
                f'await bot.pathfinder.goto(new GoalNear({bx}, {by}, {bz}, 2));',
                f'Back at base ({bx}, {by}, {bz}).', 'I returned to base.',
                mock=mock(position=[bx + offset, by, bz]),
                expect={'type': 'goto', 'target': [bx, by, bz], 'within': 2}))
    for name in ('Steve', 'Alex', 'Gabriel', 'Builder_2'):
        out.append(Trajectory(
            'navigation', f'Track {name} continuously with a non-blocking goal.',
            'setGoal keeps pathfinding active while the event loop still handles chat and damage.',
            f"bot.pathfinder.setGoal(new GoalFollow(bot.players['{name}'].entity, 3));",
            f'Following {name} in the background.', f'I am tracking {name}.',
            mock=mock(entities=[_entity(name, 6, 64, 6, 'player')]),
            expect={'type': 'setGoal', 'follow': name}))
    for x, z in ((40, 40), (-40, 40), (40, -40), (-40, -40)):
        out.append(Trajectory(
            'navigation', f'Escape the lava field by moving to x={x}, z={z}.',
            'Lava is lethal; a distant XZ target is safer than micro-managing every step.',
            f'await bot.pathfinder.goto(new GoalXZ({x}, {z}));',
            f'Cleared the lava field at x={x}, z={z}.', 'I escaped the hazard.',
            mock=mock(), expect={'type': 'goto', 'target': [x, None, z], 'xz': True}))
        out.append(Trajectory(
            'navigation', f'Walk to the village bell at ({x}, 68, {z}).',
            'Villagers trade; approach the bell so the meeting point is findable.',
            f'await bot.pathfinder.goto(new GoalNear({x}, 68, {z}, 2));',
            f'Reached the village bell at ({x}, 68, {z}).', 'I am at the village.',
            mock=mock(), expect={'type': 'goto', 'target': [x, 68, z], 'within': 2}))
    return out


def _mining(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    ores = [('diamond_ore', 50, 30, 75, 'diamond_pickaxe'),
            ('iron_ore', 12, 42, -8, 'stone_pickaxe'),
            ('coal_ore', -20, 48, 33, 'wooden_pickaxe'),
            ('gold_ore', 77, 28, 14, 'iron_pickaxe'),
            ('redstone_ore', 5, 16, 5, 'iron_pickaxe'),
            ('emerald_ore', 144, 70, -22, 'iron_pickaxe'),
            ('ancient_debris', 8, 14, -60, 'diamond_pickaxe'),
            ('obsidian', -3, 12, 9, 'diamond_pickaxe'),
            ('oak_log', 30, 66, 30, 'stone_axe'),
            ('stone', 0, 60, -4, 'wooden_pickaxe')]
    for name, x, y, z, tool in ores:
        out.append(Trajectory(
            'mining', f'Mine the {name.replace("_", " ")} at ({x}, {y}, {z}).',
            f'{tool.replace("_", " ")} is the right tool; equip it before digging so the block drops.',
            f'{_equip_hand(tool)}\nawait bot.dig(bot.blockAt(new Vec3({x}, {y}, {z})));',
            f'Mined 1 {name}.', f'I mined a {name.replace("_", " ")}.',
            mock=mock(world={_key(x, y, z): name}, inventory=[tool]),
            expect={'type': 'dig', 'block': name}))
        out.append(Trajectory(
            'mining', f'Check that ({x}, {y}, {z}) really contains {name}, then mine it.',
            'Never dig blind: read the block first and bail out if it is not the requested ore.',
            f'const block = bot.blockAt(new Vec3({x}, {y}, {z}));\n'
            f"if (!block || block.name !== '{name}') throw new Error('expected {name}');\n"
            f'{_equip_hand(tool)}\nawait bot.dig(block);',
            f'Mined 1 {name} after confirming the block.', f'Confirmed and mined the {name}.',
            mock=mock(world={_key(x, y, z): name}, inventory=[tool]),
            expect={'type': 'dig', 'block': name}))
        out.append(Trajectory(
            'mining', f'Find the nearest {name.replace("_", " ")} within 32 blocks and mine it.',
            'Searching is player-like: scan loaded blocks near me instead of assuming fixed coordinates.',
            f"const ore = bot.findBlock({{ matching: block => block.name === '{name}', maxDistance: 32 }});\n"
            f"if (!ore) throw new Error('no {name} nearby');\n"
            f'await bot.pathfinder.goto(new GoalNear(ore.position.x, ore.position.y, ore.position.z, 1));\n'
            f'{_equip_hand(tool)}\nawait bot.dig(ore);',
            f'Mined 1 {name} from the nearest vein.', f'I found and mined a {name}.',
            mock=mock(position=[x, y, z], world={_key(x, y, z): name}, inventory=[tool]),
            expect={'type': 'dig', 'block': name}))
        out.append(Trajectory(
            'mining', f'Collect three blocks of {name.replace("_", " ")} around ({x}, {y}, {z}).',
            'The collectblock plugin handles walking and digging for a batch of blocks.',
            f"const ore = bot.findBlock({{ matching: block => block.name === '{name}', maxDistance: 32 }});\n"
            f"if (!ore) throw new Error('nothing to collect');\n"
            f'await bot.collectBlock.collect(ore);',
            f'Collected 1 {name}; repeat until three are in the inventory.',
            f'I collected {name.replace("_", " ")}.',
            mock=mock(position=[x, y, z], world={_key(x, y, z): name}, inventory=[tool]),
            expect={'type': 'collect', 'block': name}))
        out.append(Trajectory(
            'mining', f'How long will mining {name} take at ({x}, {y}, {z})?',
            'digTime lets me decide whether to keep mining or switch tasks before night.',
            f'const block = bot.blockAt(new Vec3({x}, {y}, {z}));\n'
            f'const ms = bot.digTime(block);\n'
            f'bot.chat(`Mining {name} will take about ${{Math.round(ms / 1000)}}s`);',
            f'digTime reported {name} requires tool-dependent time.', 'I checked the mining time.',
            mock=mock(world={_key(x, y, z): name}, inventory=[tool]),
            expect={'type': 'chat', 'contains': name}))
    for depth in (-59, -54, -12, 12, 48):
        for direction, (dx, dz) in (('north', (0, -1)), ('south', (0, 1)),
                                    ('east', (1, 0)), ('west', (-1, 0))):
            coords = [(i * dx, depth, i * dz) for i in range(3)]
            world = {_key(*c): 'stone' for c in coords}
            digs = '\n'.join(f'await bot.dig(bot.blockAt(new Vec3({x}, {y}, {z})));'
                             for x, y, z in coords)
            out.append(Trajectory(
                'mining', f'Drive a 3-block tunnel {direction} at y={depth} from (0, {depth}, 0).',
                'A straight tunnel with a fixed height keeps the branch mine tidy and avoids caves.',
                f'{_equip_hand("iron_pickaxe")}\n{digs}',
                f'Tunnel advanced 3 blocks {direction}.', 'I extended the mine tunnel.',
                mock=mock(world=world, inventory=['iron_pickaxe']),
                expect={'type': 'dig', 'block': 'stone'}))
    for y in (11, 12, 13, 14, 15):
        out.append(Trajectory(
            'mining', f'Dig down two blocks from y={y} to reach the deepslate layer.',
            'Never dig straight down more than one block without checking what is below.',
            f"const below = bot.blockAt(new Vec3(0, {y - 1}, 0));\n"
            f"if (!below || below.name === 'lava') throw new Error('unsafe to dig down');\n"
            f'await bot.dig(below);\n'
            f'await bot.dig(bot.blockAt(new Vec3(0, {y - 2}, 0)));',
            f'Deepslate reached at y={y - 2}.', 'I dug down safely.',
            mock=mock(world={_key(0, y - 1, 0): 'stone', _key(0, y - 2, 0): 'stone'},
                      inventory=['iron_pickaxe']),
            expect={'type': 'dig', 'block': 'stone'}))
    return out


def _crafting(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    recipes = [('oak_planks', 4, False), ('stick', 4, False), ('crafting_table', 1, False),
               ('chest', 1, True), ('torch', 4, False), ('furnace', 1, True),
               ('wooden_pickaxe', 1, True), ('stone_pickaxe', 1, True),
               ('iron_sword', 1, True), ('bread', 1, True), ('ladder', 3, True),
               ('bucket', 1, True)]
    for item, count, table in recipes:
        table_arg = 'craftingTable' if table else 'null'
        setup = (f'const craftingTable = bot.blockAt(new Vec3(1, 64, 1));\n' if table else '')
        out.append(Trajectory(
            'crafting', f'Craft {count} {item.replace("_", " ")}' + (' using a crafting table.' if table else ' in the inventory grid.'),
            'Check the recipe list first; crafting blind wastes materials.' +
            (' A 3x3 recipe needs a crafting table.' if table else ' This recipe fits the 2x2 grid.'),
            f"{setup}const recipe = bot.recipesFor(bot.registry.itemsByName['{item}'].id, null, {count}, {table_arg})[0];\n"
            f"if (!recipe) throw new Error('missing materials for {item}');\n"
            f'await bot.craft(recipe, {count}, {table_arg});',
            f'Crafted {count} {item}.', f'I crafted {count} {item.replace("_", " ")}.',
            mock=mock(world={_key(1, 64, 1): 'crafting_table'}, inventory=['oak_planks', 'stick']),
            expect={'type': 'craft', 'item': item, 'count': count}))
        out.append(Trajectory(
            'crafting', f'I have no {item.replace("_", " ")}, so make one from what is in my inventory.',
            'Count the materials before starting; if something is missing, gather it instead of failing silently.',
            f"{setup}const recipe = bot.recipesAll(bot.registry.itemsByName['{item}'].id, null, {table_arg})[0];\n"
            f"if (!recipe) {{ bot.chat('I need more materials for {item}'); }} else {{ await bot.craft(recipe, 1, {table_arg}); }}",
            f'Crafting step finished for {item}.', f'I tried to craft a {item.replace("_", " ")}.',
            mock=mock(world={_key(1, 64, 1): 'crafting_table'}, inventory=['oak_planks', 'stick']),
            expect={'type': 'craft', 'item': item, 'count': 1}))
    for item, count, table in recipes[:6]:
        table_arg = 'craftingTable' if table else 'null'
        setup = (f'const craftingTable = bot.blockAt(new Vec3(1, 64, 1));\n' if table else '')
        out.append(Trajectory(
            'crafting', f'Craft {count} {item.replace("_", " ")} and tell the team what was produced.',
            'The team needs accurate inventory news; report the item only after the craft promise resolves.',
            f"{setup}const recipe = bot.recipesFor(bot.registry.itemsByName['{item}'].id, null, {count}, {table_arg})[0];\n"
            f"if (!recipe) throw new Error('no recipe for {item}');\n"
            f'await bot.craft(recipe, {count}, {table_arg});\n'
            f"bot.chat('Crafted {count} {item}');",
            f'Crafted {count} {item} and announced it.', f'I crafted {count} {item.replace("_", " ")}.',
            mock=mock(world={_key(1, 64, 1): 'crafting_table'}, inventory=['oak_planks', 'stick']),
            expect={'type': 'craft', 'item': item, 'count': count}))
    for item, count, _ in recipes[:5]:
        out.append(Trajectory(
            'crafting', f'How many {item.replace("_", " ")} can my current materials make?',
            'Ask the recipe API with minResultCount instead of guessing from memory.',
            f"const recipes = bot.recipesFor(bot.registry.itemsByName['{item}'].id, null, 1, null);\n"
            f"bot.chat(`I can craft ${{recipes.length}} recipe option(s) for {item}`);",
            f'Recipe count reported for {item}.', f'I checked what I can craft.',
            mock=mock(inventory=['oak_planks']), expect={'type': 'chat', 'contains': item}))
    extra_recipes = [('wooden_axe', 1, True), ('stone_axe', 1, True), ('iron_axe', 1, True),
                     ('wooden_shovel', 1, True), ('stone_shovel', 1, True), ('iron_shovel', 1, True),
                     ('iron_hoe', 1, True), ('shears', 1, True), ('bowl', 4, True),
                     ('cake', 1, True), ('iron_helmet', 1, True), ('iron_chestplate', 1, True),
                     ('iron_leggings', 1, True), ('iron_boots', 1, True), ('shield', 1, True)]
    for item, count, table in extra_recipes:
        table_arg = 'craftingTable' if table else 'null'
        setup = (f'const craftingTable = bot.blockAt(new Vec3(1, 64, 1));\n' if table else '')
        for announce in (False, True):
            tail = f"\nbot.chat('Crafted {item}');" if announce else ''
            out.append(Trajectory(
                'crafting', f'Craft {count} {item.replace("_", " ")} for the next expedition.',
                'Check the recipe, craft exactly what is needed, then report the result.',
                f"{setup}const recipe = bot.recipesFor(bot.registry.itemsByName['{item}'].id, null, {count}, {table_arg})[0];\n"
                f"if (!recipe) throw new Error('no recipe for {item}');\n"
                f'await bot.craft(recipe, {count}, {table_arg});{tail}',
                f'Crafted {count} {item}.', f'I crafted {count} {item.replace("_", " ")}.',
                mock=mock(world={_key(1, 64, 1): 'crafting_table'}, inventory=['oak_planks', 'stick', 'coal']),
                expect={'type': 'craft', 'item': item, 'count': count}))
    for item, count in (('oak_planks', 8), ('stick', 8), ('torch', 16), ('ladder', 6),
                        ('crafting_table', 2), ('chest', 2)):
        table = item in ('ladder', 'chest')
        table_arg = 'craftingTable' if table else 'null'
        setup = (f'const craftingTable = bot.blockAt(new Vec3(1, 64, 1));\n' if table else '')
        out.append(Trajectory(
            'crafting', f'Craft a batch of {count} {item.replace("_", " ")} for the base.',
            'Batching a known recipe avoids repeated trips to the crafting table.',
            f"{setup}const recipe = bot.recipesFor(bot.registry.itemsByName['{item}'].id, null, {count}, {table_arg})[0];\n"
            f"if (!recipe) throw new Error('no recipe for {item}');\n"
            f'await bot.craft(recipe, {count}, {table_arg});',
            f'Crafted {count} {item}.', f'I finished the {item.replace("_", " ")} batch.',
            mock=mock(world={_key(1, 64, 1): 'crafting_table'}, inventory=['oak_planks', 'stick']),
            expect={'type': 'craft', 'item': item, 'count': count}))
    return out


def _combat(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    mobs = ['zombie', 'skeleton', 'creeper', 'spider', 'enderman', 'witch', 'pillager', 'husk']
    weapons = ['iron_sword', 'diamond_sword', 'stone_sword']
    for i, mob in enumerate(mobs):
        weapon = weapons[i % len(weapons)]
        ent = _entity(mob, 3, 64, 2)
        out.append(Trajectory(
            'combat', f'There is a {mob} nearby. Defend the base.',
            f'A hostile {mob} is within range; equip the best sword, then attack rather than flee.',
            f'{_find_entity(mob)}\n'
            f"if (!target) throw new Error('no {mob} in range');\n"
            f'{_equip_hand(weapon)}\nawait bot.attack(target);',
            f'Attacked {mob}, dealt 5 damage.', f'I attacked the {mob}.',
            mock=mock(entities=[ent], inventory=[weapon]),
            expect={'type': 'attack', 'entity': mob}))
        out.append(Trajectory(
            'combat', f'Fight off the {mob} until it dies.',
            'One hit is not enough; keep swinging while checking that the target is still alive.',
            f'{_find_entity(mob)}\n'
            f"if (!target) throw new Error('no {mob} in range');\n"
            f'{_equip_hand(weapon)}\n'
            f"for (let i = 0; i < 5 && target.isValid; i++) {{ await bot.attack(target); }}",
            f'{mob.capitalize()} defeated.', f'The {mob} is dead.',
            mock=mock(entities=[ent], inventory=[weapon]),
            expect={'type': 'attack', 'entity': mob, 'min_count': 1}))
    for mob in mobs[:5]:
        ent = _entity(mob, 2, 64, 1)
        out.append(Trajectory(
            'combat', f'A {mob} is blocking the corridor. Clear it without destroying the base.',
            'Use the sword, not blocks; pathfinder keeps the bot at reachable melee distance.',
            f'{_find_entity(mob, 8)}\n'
            f"if (!target) throw new Error('no {mob} in range');\n"
            f'await bot.pathfinder.goto(new GoalNear(target.position.x, target.position.y, target.position.z, 2));\n'
            f'{_equip_hand("iron_sword")}\nawait bot.attack(target);',
            f'{mob.capitalize()} hit and pushed back.', f'I engaged the {mob}.',
            mock=mock(entities=[ent], inventory=['iron_sword']),
            expect={'type': 'attack', 'entity': mob}))
    for mob in mobs[:5]:
        ent = _entity(mob, 4, 64, 0)
        out.append(Trajectory(
            'combat', f'The {mob} is too strong and health is low. Retreat.',
            'Survival is a valid player policy: disengage, eat, and come back with better gear.',
            f'bot.pathfinder.setGoal(new GoalXZ(bot.entity.position.x + 24, bot.entity.position.z + 24));\n'
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['bread'].id, null), 'hand');\n"
            f'await bot.consume();',
            'Retreated 24 blocks and ate food.', 'I backed off to recover.',
            mock=mock(entities=[ent], inventory=['bread'], food=6),
            expect={'type': 'consume'}))
    for player in ('Steve', 'Alex'):
        ent = _entity(player, 2, 64, 2, 'player')
        out.append(Trajectory(
            'combat', f'{player} is duelling me in the arena. Attack their entity.',
            'PvP plugin attack handles cooldown and positioning for a player target.',
            f"bot.pvp.attack(bot.players['{player}'].entity);",
            f'Attacking {player}.', f'I attacked {player}.',
            mock=mock(entities=[ent], inventory=['diamond_sword']),
            expect={'type': 'pvp_attack', 'entity': player}))
    for hp in (6, 8, 10):
        out.append(Trajectory(
            'combat', f'Health is {hp}/20 after the fight. Eat before the next mob.',
            'Hunger regenerates health; eat to full before pulling another mob.',
            f"const food = bot.inventory.findInventoryItem(bot.registry.itemsByName['cooked_beef'].id, null);\n"
            f"if (!food) throw new Error('no food');\n"
            f'await bot.equip(food, "hand");\nawait bot.consume();',
            'Health regenerating after eating.', 'I ate to recover health.',
            mock=mock(inventory=['cooked_beef'], health=hp, food=6),
            expect={'type': 'consume'}))
    waves = [('zombie', 'skeleton'), ('spider', 'creeper'), ('husk', 'pillager'),
             ('enderman', 'witch')]
    for first, second in waves:
        for health in (20, 14):
            out.append(Trajectory(
                'combat', f'A {first} and a {second} are attacking the base at night.',
                'Take the nearest hostile first, then retarget; do not chase a fleeing mob into the dark.',
                f'{_find_entity(first)}\n'
                f"if (!target) throw new Error('no {first} in range');\n"
                f'{_equip_hand("iron_sword")}\n'
                f'await bot.attack(target);\n'
                f'const next = bot.nearestEntity(e => e.type === \'mob\' && e.name === \'{second}\');\n'
                f'if (next) await bot.attack(next);',
                f'Fought the {first} first and then the {second}.', 'The base is clear.',
                mock=mock(health=health, inventory=['iron_sword'],
                          entities=[_entity(first, 2, 64, 1), _entity(second, 4, 64, 3)]),
                expect={'type': 'attack', 'entity': first}))
    for mob in ('zombie', 'skeleton', 'creeper', 'spider', 'husk'):
        out.append(Trajectory(
            'combat', f'Kite the {mob}: hit it, step back, hit again.',
            'Kiting reduces incoming damage when armour is weak; keep the target in reach.',
            f'{_find_entity(mob)}\n'
            f"if (!target) throw new Error('no {mob} in range');\n"
            f'{_equip_hand("iron_sword")}\n'
            f'for (let i = 0; i < 3; i++) {{ await bot.attack(target); await bot.pathfinder.goto(new GoalNear(bot.entity.position.x - 2, bot.entity.position.y, bot.entity.position.z, 1)); }}',
            f'Kited and hit the {mob} three times.', f'I kited the {mob}.',
            mock=mock(inventory=['iron_sword'], entities=[_entity(mob, 2, 64, 1)]),
            expect={'type': 'attack', 'entity': mob}))
    for mob in ('zombie', 'skeleton', 'creeper', 'enderman', 'witch'):
        out.append(Trajectory(
            'combat', f'Shoot the {mob} from the battlement before it reaches the door.',
            'A bow is safer at range; draw, release, then switch back to the sword.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['bow'].id, null), 'hand');\n"
            f'bot.activateItem();\n'
            f'await bot.deactivateItem();',
            f'Loosed an arrow at the {mob}.', f'I shot at the {mob}.',
            mock=mock(inventory=['bow'], entities=[_entity(mob, 8, 64, 0)]),
            expect={'type': 'activateItem'}))
    for slot, item in (('head', 'iron_helmet'), ('torso', 'iron_chestplate'),
                       ('legs', 'iron_leggings'), ('feet', 'iron_boots')):
        out.append(Trajectory(
            'combat', f'A raid is coming. Put on the {item.replace("_", " ")}.',
            'Armour first, weapon second: the damage reduction matters more than one sword hit.',
            f"const piece = bot.inventory.findInventoryItem(bot.registry.itemsByName['{item}'].id, null);\n"
            f"if (!piece) throw new Error('no {item}');\nawait bot.equip(piece, '{slot}');\n"
            f'{_equip_hand("diamond_sword")}',
            f'Equipped {item} and the sword.', 'I am ready for the raid.',
            mock=mock(inventory=[item, 'diamond_sword']),
            expect={'type': 'equip', 'item': item, 'destination': slot}))
    for hp in (6, 8, 10):
        out.append(Trajectory(
            'combat', f'Health is {hp}/20 mid-fight. Decide whether to eat or flee.',
            'Below 10 HP the risk is too high: eat, then disengage if the mob is still alive.',
            f"if (bot.health < 12) {{ await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['cooked_beef'].id, null), 'hand'); await bot.consume(); }}",
            'Ate and reassessed the fight.', 'I healed before continuing.',
            mock=mock(inventory=['cooked_beef'], health=hp), expect={'type': 'consume'}))
    return out


def _building(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    materials = ['cobblestone', 'oak_planks', 'stone', 'dirt']
    for mat in materials:
        for length in (3, 5, 7):
            x0 = -length // 2
            out.append(Trajectory(
                'building', f'Build a {length}-block {mat.replace("_", " ")} wall along x at y=64, z=0.',
                'Equip the material once, then place blocks in a deterministic line; do not place inside myself.',
                f"const block = bot.inventory.findInventoryItem(bot.registry.itemsByName['{mat}'].id, null);\n"
                f"if (!block) throw new Error('no {mat}');\nawait bot.equip(block, 'hand');\n"
                f'for (let x = {x0}; x < {x0 + length}; x++) {{ await bot.placeBlock(bot.blockAt(new Vec3(x, 63, 0)), new Vec3(0, 1, 0)); }}',
                f'Placed {length} {mat} blocks.', f'The wall is {length} blocks long.',
                mock=mock(world={_key(x, 63, 0): 'stone' for x in range(x0, x0 + length)}, inventory=[mat]),
                expect={'type': 'place', 'block': mat, 'min_count': length}))
    for height in (2, 3, 4, 5):
        out.append(Trajectory(
            'building', f'Pillar up {height} blocks to reach a ledge at y={64 + height}.',
            'Jump-and-place is the standard pillar: look down, jump, place under the feet, repeat.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['cobblestone'].id, null), 'hand');\n"
            f'for (let i = 0; i < {height}; i++) {{ bot.setControlState("jump", true); await bot.placeBlock(bot.blockAt(bot.entity.position.offset(0, -1, 0)), new Vec3(0, 1, 0)); bot.setControlState("jump", false); }}',
            f'Pillared up {height} blocks.', f'I climbed {height} blocks.',
            mock=mock(world={_key(0, 63, 0): 'stone'}, inventory=['cobblestone']),
            expect={'type': 'place', 'block': 'cobblestone', 'min_count': height}))
    for width in (3, 4):
        out.append(Trajectory(
            'building', f'Build a {width}x{width} floor of oak planks at y=64.',
            'A floor is a nested loop; place against a fixed reference block so faces are predictable.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['oak_planks'].id, null), 'hand');\n"
            f'for (let x = 0; x < {width}; x++) {{ for (let z = 0; z < {width}; z++) {{ await bot.placeBlock(bot.blockAt(new Vec3(x, 63, z)), new Vec3(0, 1, 0)); }} }}',
            f'Placed {width * width} oak planks.', f'The floor is {width * width} blocks.',
            mock=mock(world={_key(x, 63, z): 'stone' for x in range(width) for z in range(width)}, inventory=['oak_planks']),
            expect={'type': 'place', 'block': 'oak_planks', 'min_count': width * width}))
    # The user's "build a house" example, made executable against the mock world.
    out.append(Trajectory(
        'building', 'Build a 7x7 wooden house with 3-block walls and a flat roof.',
        'Clear a footprint, raise the walls, then cap the roof; check materials before starting.',
        "await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['oak_planks'].id, null), 'hand');\n"
        'for (let y = 64; y < 67; y++) {\n'
        '  for (let x = -3; x <= 3; x++) {\n'
        '    for (let z = -3; z <= 3; z++) {\n'
        '      if (Math.abs(x) === 3 || Math.abs(z) === 3) {\n'
        '        await bot.placeBlock(bot.blockAt(new Vec3(x, y - 1, z)), new Vec3(0, 1, 0));\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
        'for (let x = -3; x <= 3; x++) {\n'
        '  for (let z = -3; z <= 3; z++) {\n'
        '    const top = bot.blockAt(new Vec3(x, 66, z));\n'
        "    if (top.name !== 'air') await bot.placeBlock(top, new Vec3(0, 1, 0));\n"
        '  }\n'
        '}\n'
        'for (let x = -2; x <= 2; x++) {\n'
        '  for (let z = -2; z <= 2; z++) {\n'
        '    await bot.placeBlock(bot.blockAt(new Vec3(x - 1, 67, z)), new Vec3(1, 0, 0));\n'
        '  }\n'
        '}',
        'House shell built: 4 walls x 3 high plus a 7x7 roof.', 'The house is finished.',
        mock=mock(world={_key(x, 63, z): 'stone' for x in range(-3, 4) for z in range(-3, 4)},
                 inventory=['oak_planks']),
        expect={'type': 'place', 'block': 'oak_planks', 'min_count': 80}))
    for count, spacing in ((4, 4), (6, 3)):
        out.append(Trajectory(
            'building', f'Place {count} torches every {spacing} blocks along the mine tunnel.',
            'Light prevents hostile spawns; keep the torch in hand and place on the wall face.',
            f"const torch = bot.inventory.findInventoryItem(bot.registry.itemsByName['torch'].id, null);\n"
            f"if (!torch) throw new Error('no torches');\nawait bot.equip(torch, 'hand');\n"
            f'for (let i = 0; i < {count}; i++) {{ await bot.placeBlock(bot.blockAt(new Vec3(i * {spacing}, 64, 2)), new Vec3(0, 0, -1)); }}',
            f'Placed {count} torches in the tunnel.', f'I lit the tunnel with {count} torches.',
            mock=mock(world={_key(i * spacing, 64, 2): 'stone' for i in range(count)}, inventory=['torch']),
            expect={'type': 'place', 'block': 'torch', 'min_count': count}))
    for mat in ('cobblestone', 'oak_planks', 'stone', 'dirt'):
        for length in (2, 4, 6, 8, 10):
            x0 = -length // 2
            out.append(Trajectory(
                'building', f'Extend the {mat.replace("_", " ")} wall to {length} blocks.',
                'Place from one end to the other so each block has a solid neighbour.',
                f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['{mat}'].id, null), 'hand');\n"
                f'for (let x = {x0}; x < {x0 + length}; x++) {{ await bot.placeBlock(bot.blockAt(new Vec3(x, 63, 0)), new Vec3(0, 1, 0)); }}',
                f'Wall now spans {length} blocks.', f'The wall is {length} long.',
                mock=mock(world={_key(x, 63, 0): 'stone' for x in range(x0, x0 + length)}, inventory=[mat]),
                expect={'type': 'place', 'block': mat, 'min_count': length}))
    for size in (3, 5, 6, 7):
        out.append(Trajectory(
            'building', f'Lay a {size}x{size} cobblestone floor.',
            'A nested loop over x and z is the simplest reliable floor pattern.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['cobblestone'].id, null), 'hand');\n"
            f'for (let x = 0; x < {size}; x++) {{ for (let z = 0; z < {size}; z++) {{ await bot.placeBlock(bot.blockAt(new Vec3(x, 63, z)), new Vec3(0, 1, 0)); }} }}',
            f'Floor complete: {size * size} blocks.', f'The floor is {size}x{size}.',
            mock=mock(world={_key(x, 63, z): 'stone' for x in range(size) for z in range(size)},
                      inventory=['cobblestone']),
            expect={'type': 'place', 'block': 'cobblestone', 'min_count': size * size}))
    for height in (3, 4, 5):
        out.append(Trajectory(
            'building', f'Climb a {height}-block cobblestone staircase.',
            'Step up one block at a time; the reference block moves with the staircase.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['cobblestone'].id, null), 'hand');\n"
            f'for (let i = 0; i < {height}; i++) {{ await bot.placeBlock(bot.blockAt(new Vec3(i, 63 + i, 0)), new Vec3(0, 1, 0)); }}',
            f'Staircase of {height} steps built.', 'The stairs are ready.',
            mock=mock(world={_key(i, 63 + i, 0): 'stone' for i in range(height)}, inventory=['cobblestone']),
            expect={'type': 'place', 'block': 'cobblestone', 'min_count': height}))
    for doorway in ('oak_planks', 'cobblestone', 'stone'):
        out.append(Trajectory(
            'building', f'Frame a 2x3 doorway in the {doorway.replace("_", " ")} wall.',
            'Leave the middle two columns open so players can walk through.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['{doorway}'].id, null), 'hand');\n"
            f'for (const x of [0, 1, 3, 4]) {{ await bot.placeBlock(bot.blockAt(new Vec3(x, 63, 0)), new Vec3(0, 1, 0)); }}',
            'Door frame placed with a two-block opening.', 'The doorway is framed.',
            mock=mock(world={_key(x, 63, 0): 'stone' for x in (0, 1, 3, 4)}, inventory=[doorway]),
            expect={'type': 'place', 'block': doorway, 'min_count': 4}))
    for count in (3, 5, 7):
        out.append(Trajectory(
            'building', f'Light the {count}-block corridor with torches on the ceiling.',
            'Ceiling torches keep the floor walkable; one torch roughly every four blocks.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['torch'].id, null), 'hand');\n"
            f'for (let i = 0; i < {count}; i++) {{ await bot.placeBlock(bot.blockAt(new Vec3(i * 4, 66, 0)), new Vec3(0, -1, 0)); }}',
            f'Placed {count} ceiling torches.', 'The corridor is lit.',
            mock=mock(world={_key(i * 4, 66, 0): 'stone' for i in range(count)}, inventory=['torch']),
            expect={'type': 'place', 'block': 'torch', 'min_count': count}))
    return out


def _inventory(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    armour = [('iron_helmet', 'head'), ('iron_chestplate', 'torso'),
              ('iron_leggings', 'legs'), ('iron_boots', 'feet')]
    for item, slot in armour:
        out.append(Trajectory(
            'inventory', f'Equip the {item.replace("_", " ")}.',
            f'Armour goes in the {slot} slot; equip only an item that actually exists in the inventory.',
            f"const piece = bot.inventory.findInventoryItem(bot.registry.itemsByName['{item}'].id, null);\n"
            f"if (!piece) throw new Error('no {item}');\nawait bot.equip(piece, '{slot}');",
            f'Equipped {item}.', f'I equipped the {item.replace("_", " ")}.',
            mock=mock(inventory=[item]), expect={'type': 'equip', 'item': item, 'destination': slot}))
    for item, count in (('dirt', 32), ('cobblestone', 64), ('gravel', 16), ('wheat_seeds', 12)):
        out.append(Trajectory(
            'inventory', f'Drop {count} {item.replace("_", " ")} to free inventory space.',
            'Keep tools and food; drop the bulk junk stack instead of everything.',
            f'await bot.toss(bot.registry.itemsByName["{item}"].id, null, {count});',
            f'Dropped {count} {item}.', f'I dropped {count} {item.replace("_", " ")}.',
            mock=mock(inventory=[item]), expect={'type': 'toss', 'item': item, 'count': count}))
    for food in ('bread', 'apple', 'cooked_beef', 'golden_apple'):
        out.append(Trajectory(
            'inventory', f'My hunger is low. Eat the {food.replace("_", " ")}.',
            'Food restores hunger, not just health; hold it in the hand before consuming.',
            f"const food = bot.inventory.findInventoryItem(bot.registry.itemsByName['{food}'].id, null);\n"
            f"if (!food) throw new Error('no {food}');\nawait bot.equip(food, 'hand');\nawait bot.consume();",
            f'Ate 1 {food}.', f'I ate a {food.replace("_", " ")}.',
            mock=mock(inventory=[food], food=8), expect={'type': 'consume', 'item': food}))
    for slot_x, slot_z in ((1, 1), (4, 1), (-2, 3)):
        out.append(Trajectory(
            'inventory', f'Deposit 16 cobblestone into the chest at ({slot_x}, 64, {slot_z}).',
            'Walk to the chest, open it, deposit only the requested amount, then close it.',
            f'const chestBlock = bot.blockAt(new Vec3({slot_x}, 64, {slot_z}));\n'
            f"if (!chestBlock) throw new Error('chest is not loaded');\n"
            f'const chest = await bot.openContainer(chestBlock);\n'
            f"await chest.deposit(bot.registry.itemsByName['cobblestone'].id, null, 16);\n"
            f'chest.close();',
            'Deposited 16 cobblestone and closed the chest.', 'I stored the cobblestone.',
            mock=mock(world={_key(slot_x, 64, slot_z): 'chest'}, inventory=['cobblestone']),
            expect={'type': 'deposit', 'item': 'cobblestone', 'count': 16}))
    for x, z in ((1, 1), (4, 1)):
        out.append(Trajectory(
            'inventory', f'Take 8 bread out of the chest at ({x}, 64, {z}).',
            'Withdraw a bounded amount so the chest keeps supplies for other players.',
            f'const chest = await bot.openContainer(bot.blockAt(new Vec3({x}, 64, {z})));\n'
            f"await chest.withdraw(bot.registry.itemsByName['bread'].id, null, 8);\n"
            f'chest.close();',
            'Withdrew 8 bread.', 'I took bread from the chest.',
            mock=mock(world={_key(x, 64, z): 'chest'}, inventory=['bread']),
            expect={'type': 'withdraw', 'item': 'bread', 'count': 8}))
    counts = [('diamond', 12), ('oak_log', 34), ('torch', 5)]
    for item, _ in counts:
        out.append(Trajectory(
            'inventory', f'Count how many {item.replace("_", " ")} I am carrying.',
            'Sum the stack counts rather than reporting the number of stacks.',
            f"const total = bot.inventory.items().filter(i => i.name === '{item}').reduce((n, i) => n + i.count, 0);\n"
            f"bot.chat(`I have ${{total}} {item}`);",
            f'Reported the {item} total.', f'I counted my {item.replace("_", " ")}.',
            mock=mock(inventory=[item]), expect={'type': 'chat', 'contains': item}))
    for item in ('iron_sword', 'iron_pickaxe', 'torch'):
        out.append(Trajectory(
            'inventory', f'Put the {item.replace("_", " ")} in my hand before the next task.',
            'Hotbar selection matters: equip explicitly instead of assuming the held item.',
            f'await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName["{item}"].id, null), "hand");',
            f'Holding {item}.', f'I equipped the {item.replace("_", " ")}.',
            mock=mock(inventory=[item]), expect={'type': 'equip', 'item': item, 'destination': 'hand'}))
    out.append(Trajectory(
        'inventory', 'Check how many inventory slots are still free.',
        'Stopping to check capacity avoids dropping items on the ground mid-task.',
        "const free = bot.inventory.emptySlotCount();\nbot.chat(`I have ${free} free slots`);",
        'Reported free slot count.', 'I checked my inventory space.',
        mock=mock(inventory=['cobblestone']), expect={'type': 'chat', 'contains': 'free slots'}))
    for item in ('iron_pickaxe', 'stone_axe', 'iron_shovel', 'wooden_hoe', 'shears'):
        for hand in ('hand',):
            out.append(Trajectory(
                'inventory', f'Select the {item.replace("_", " ")} for the next job.',
                'Explicitly equipping the tool avoids breaking blocks with the wrong item.',
                f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['{item}'].id, null), '{hand}');",
                f'Holding {item}.', f'I equipped the {item.replace("_", " ")}.',
                mock=mock(inventory=[item]),
                expect={'type': 'equip', 'item': item, 'destination': hand}))
    for item, count in (('gravel', 8), ('dirt', 16), ('cobblestone', 32),
                        ('sand', 64), ('wheat_seeds', 4)):
        out.append(Trajectory(
            'inventory', f'The inventory is full. Drop {count} {item.replace("_", " ")}.',
            'Free slots before mining; keep one stack of building material.',
            f'await bot.toss(bot.registry.itemsByName["{item}"].id, null, {count});',
            f'Dropped {count} {item}.', f'I cleared {count} {item.replace("_", " ")}.',
            mock=mock(inventory=[item]), expect={'type': 'toss', 'item': item, 'count': count}))
    for item in ('diamond', 'iron_ingot', 'coal', 'redstone', 'oak_log', 'torch', 'bread', 'arrow'):
        out.append(Trajectory(
            'inventory', f'Count my {item.replace("_", " ")} before the trip.',
            'Know the stock before crafting; report a number, not a guess.',
            f"const total = bot.inventory.items().filter(i => i.name === '{item}').reduce((n, i) => n + i.count, 0);\n"
            f"bot.chat(`I have ${{total}} {item}`);",
            f'Counted the {item} stacks.', f'I know how much {item.replace("_", " ")} I have.',
            mock=mock(inventory=[item]), expect={'type': 'chat', 'contains': item}))
    for item, count in (('cobblestone', 8), ('oak_planks', 16), ('torch', 4), ('bread', 2)):
        out.append(Trajectory(
            'inventory', f'Store {count} {item.replace("_", " ")} in the base chest.',
            'Deposit surplus at base instead of carrying everything into the mine.',
            f'const chest = await bot.openContainer(bot.blockAt(new Vec3(1, 64, 1)));\n'
            f'await chest.deposit(bot.registry.itemsByName["{item}"].id, null, {count});\nchest.close();',
            f'Stored {count} {item}.', f'I stored the {item.replace("_", " ")}.',
            mock=mock(world={_key(1, 64, 1): 'chest'}, inventory=[item]),
            expect={'type': 'deposit', 'item': item, 'count': count}))
    for item, count in (('bread', 4), ('torch', 8), ('cobblestone', 32), ('cooked_beef', 2)):
        out.append(Trajectory(
            'inventory', f'Take {count} {item.replace("_", " ")} from the base chest.',
            'Withdraw what the current task needs and leave the rest for teammates.',
            f'const chest = await bot.openContainer(bot.blockAt(new Vec3(1, 64, 1)));\n'
            f'await chest.withdraw(bot.registry.itemsByName["{item}"].id, null, {count});\nchest.close();',
            f'Withdrew {count} {item}.', f'I took the {item.replace("_", " ")}.',
            mock=mock(world={_key(1, 64, 1): 'chest'}, inventory=[item]),
            expect={'type': 'withdraw', 'item': item, 'count': count}))
    return out


def _chat(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    greetings = ['Hello everyone!', 'Oi, cheguei!', 'Good morning, team.', 'Anyone need help?']
    for text in greetings:
        out.append(Trajectory(
            'chat', f'Join the server and say: {text}',
            'A short greeting tells the other players a bot is present and listening.',
            f'bot.chat({_q(text)});', f'Chat message sent: {text}', 'I greeted the server.',
            mock=mock(), expect={'type': 'chat', 'contains': text}))
    status = [('health', 18, 'HP'), ('food', 12, 'food'), ('position', None, 'position')]
    for key, value, label in status:
        if key == 'health':
            action = 'bot.chat(`Health: ${bot.health}/20`);'
        elif key == 'food':
            action = 'bot.chat(`Food: ${bot.food}/20`);'
        else:
            action = 'bot.chat(`Position: (${bot.entity.position.x}, ${bot.entity.position.y}, ${bot.entity.position.z})`);'
        out.append(Trajectory(
            'chat', f'Report my {label} to the team.',
            'Status reports should come from live bot state, not from memory.',
            action, f'Reported my {label}.', f'I shared my {label}.',
            mock=mock(health=value, food=value if key == 'food' else 20),
            expect={'type': 'chat', 'contains': {'position': 'Position', 'health': 'Health',
                                                  'food': 'Food'}[key]}))
    for player in ('Steve', 'Alex', 'Gabriel'):
        for text in ('Come to my base, I have spare iron.', 'Do you need food?', 'I found diamonds at y=-54.'):
            out.append(Trajectory(
                'chat', f'Whisper {player}: {text}',
                'Whispering keeps trade information out of global chat.',
                f'bot.whisper({_q(player)}, {_q(text)});',
                f'Whisper delivered to {player}.', f'I messaged {player}.',
                mock=mock(entities=[_entity(player, 2, 64, 2, 'player')]),
                expect={'type': 'whisper', 'player': player, 'contains': text}))
    listeners = [
        ('chat', "bot.on('chat', (username, message) => {\n"
                 "  if (username === bot.username) return;\n"
                 "  if (message === 'come') bot.chat('On my way');\n"
                 '});', 'chat', 'come'),
        ('spawn', "bot.once('spawn', () => {\n  bot.chat('Bot online');\n});", 'spawn', 'online'),
        ('death', "bot.on('death', () => {\n  bot.chat('I died, returning to base');\n});", 'death', 'died'),
        ('kicked', "bot.on('kicked', reason => {\n  console.log('Kicked: ' + reason);\n});", 'kicked', 'reason'),
    ]
    for event, code, event_name, text in listeners:
        out.append(Trajectory(
            'chat', f'React to the {event_name} event without polling.',
            'Events are how a player-like bot reacts to the world; register the handler once.',
            code, f'Handler registered for {event_name}.', f'I can react to {event_name}.',
            mock=mock(), expect={'type': 'listen', 'event': event}))
    answers = [('Where are you?', 'I am at the base, near the chests.'),
               ('Can you bring wood?', 'Yes, I will bring oak logs.'),
               ('Is it night?', 'Yes, I am placing torches around the base.'),
               ('Need help mining?', 'Sure, I will meet you at the mine entrance.')]
    for question, reply in answers:
        out.append(Trajectory(
            'chat', f'{question} Reply in chat.',
            'Answer only what I know from the current session; do not invent coordinates.',
            f'bot.chat({_q(reply)});', f'Replied: {reply}', 'I answered in chat.',
            mock=mock(), expect={'type': 'chat', 'contains': reply[:12]}))
    for i in range(8):
        out.append(Trajectory(
            'chat', f'Send trade offer {i + 1}: 16 cobblestone for 4 bread.',
            'Trade offers need an amount and a recipient; keep chat concise.',
            f"bot.chat('Trade offer: 16 cobblestone for 4 bread, anyone?');",
            'Trade offer posted.', 'I offered a trade.',
            mock=mock(), expect={'type': 'chat', 'contains': 'Trade offer'}))
    for player in ('Steve', 'Alex', 'Gabriel', 'Builder_2', 'Miner_7', 'Farmer_3'):
        out.append(Trajectory(
            'chat', f'Greet {player} when they join the server.',
            'Greeting a specific player is friendlier than a generic broadcast.',
            f"bot.chat('Welcome back, {player}!');", f'Greeted {player}.', f'I said hello to {player}.',
            mock=mock(entities=[_entity(player, 2, 64, 2, 'player')]),
            expect={'type': 'chat', 'contains': player}))
    for x, y, z in ((100, 64, 200), (-37, 70, 12), (512, 63, -256), (0, 64, 0)):
        out.append(Trajectory(
            'chat', f'Report the current position ({x}, {y}, {z}) to the team.',
            'Coordinates let teammates find me without a map marker.',
            f'bot.chat("Position: ({x}, {y}, {z})");',
            f'Shared position ({x}, {y}, {z}).', 'I sent my coordinates.',
            mock=mock(position=[x, y, z]), expect={'type': 'chat', 'contains': f'{x}'}))
    for event, text in (('playerJoined', 'joined'), ('playerLeft', 'left'),
                        ('health', 'health'), ('rain', 'rain')):
        out.append(Trajectory(
            'chat', f'React in chat when the {event} event fires.',
            'Event handlers keep reactions immediate without polling the server.',
            f"bot.on('{event}', () => {{\n  bot.chat('Noticed {text}');\n}});",
            f'Handler registered for {event}.', f'I react to {event}.',
            mock=mock(), expect={'type': 'listen', 'event': event}))
    for question, reply in [('Are you at base?', 'Yes, sorting the chests.'),
                            ('Can you craft torches?', 'I can, I have coal and sticks.'),
                            ('Is the mine safe?', 'Bring a sword, mobs spawned at y=12.'),
                            ('Who has spare food?', 'I have bread to share.'),
                            ('What are you doing?', 'Building the north wall.'),
                            ('Need a hand?', 'Yes, bring cobblestone.')]:
        out.append(Trajectory(
            'chat', f'{question} Answer briefly.',
            'Answer from current state; never invent resources I do not have.',
            f'bot.chat({_q(reply)});', f'Answered: {reply}', 'I replied in chat.',
            mock=mock(), expect={'type': 'chat', 'contains': reply[:10]}))
    for amount in (8, 16, 24, 32, 48, 64):
        out.append(Trajectory(
            'chat', f'Offer {amount} cobblestone for trade in global chat.',
            'State the amount and what I want so the offer is actionable.',
            f"bot.chat('Selling {amount} cobblestone, looking for iron.');",
            f'Posted a {amount}-block cobblestone trade.', 'I offered cobblestone.',
            mock=mock(), expect={'type': 'chat', 'contains': f'{amount} cobblestone'}))
    return out


def _farming(rng: random.Random) -> list[Trajectory]:
    out: list[Trajectory] = []
    crops = [('wheat', 'wheat_seeds', 7), ('carrots', 'carrot', 7),
             ('potatoes', 'potatoes', 7), ('beetroot', 'beetroot_seeds', 3)]
    for crop, seed, age in crops:
        for x, z in ((0, 0), (4, 0), (8, 0)):
            out.append(Trajectory(
                'farming', f'Harvest the mature {crop} at ({x}, 64, {z}) and replant.',
                f'Only harvest age {age}; replant immediately so the farm keeps producing.',
                f"const crop = bot.findBlock({{ matching: b => b.name === '{crop}' && b.getProperties().age === {age}, maxDistance: 16 }});\n"
                f"if (!crop) throw new Error('no mature {crop}');\n"
                f'await bot.pathfinder.goto(new GoalBlock(crop.position.x, crop.position.y, crop.position.z));\n'
                f'await bot.dig(crop);\n'
                f"const seeds = bot.inventory.findInventoryItem(bot.registry.itemsByName['{seed}'].id, null);\n"
                f"if (!seeds) throw new Error('no {seed}');\n"
                f"await bot.equip(seeds, 'hand');\n"
                f'const soil = bot.blockAt(crop.position.offset(0, -1, 0));\n'
                f'await bot.placeBlock(soil, new Vec3(0, 1, 0));',
                f'Harvested {crop} and replanted {seed}.', f'I harvested and replanted the {crop}.',
                mock=mock(position=[x, 64, z],
                          world={_key(x, 64, z): crop, _key(x, 63, z): 'farmland'},
                          inventory=[seed]),
                expect={'type': 'place', 'block': seed}))
            out.append(Trajectory(
                'farming', f'Plant {seed.replace("_", " ")} on the empty farmland at ({x}, 64, {z}).',
                'Seeds need farmland below and air above; equip the seeds and place against the soil.',
                f"const seeds = bot.inventory.findInventoryItem(bot.registry.itemsByName['{seed}'].id, null);\n"
                f"if (!seeds) throw new Error('no {seed}');\nawait bot.equip(seeds, 'hand');\n"
                f'await bot.placeBlock(bot.blockAt(new Vec3({x}, 63, {z})), new Vec3(0, 1, 0));',
                f'Planted {seed} at ({x}, 64, {z}).', f'I planted {seed.replace("_", " ")}.',
                mock=mock(world={_key(x, 63, z): 'farmland'}, inventory=[seed]),
                expect={'type': 'place', 'block': seed}))
            out.append(Trajectory(
                'farming', f'Use bone meal on the {crop} at ({x}, 64, {z}).',
                'Bone meal speeds growth; hold it and activate the crop block.',
                f"const meal = bot.inventory.findInventoryItem(bot.registry.itemsByName['bone_meal'].id, null);\n"
                f"if (!meal) throw new Error('no bone meal');\nawait bot.equip(meal, 'hand');\n"
                f"await bot.activateBlock(bot.blockAt(new Vec3({x}, 64, {z})));",
                f'Applied bone meal to {crop}.', f'I fertilised the {crop}.',
                mock=mock(world={_key(x, 64, z): crop}, inventory=['bone_meal']),
                expect={'type': 'activateBlock', 'block': crop}))
    for x, z in ((2, 2), (6, 2)):
        out.append(Trajectory(
            'farming', f'Till the grass block at ({x}, 64, {z}) into farmland.',
            'A hoe converts grass to farmland; check the tool is in hand first.',
            f'await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName["iron_hoe"].id, null), "hand");\n'
            f'await bot.activateBlock(bot.blockAt(new Vec3({x}, 64, {z})));',
            'Grass block tilled into farmland.', 'I tilled the soil.',
            mock=mock(world={_key(x, 64, z): 'grass_block'}, inventory=['iron_hoe']),
            expect={'type': 'activateBlock', 'block': 'grass_block'}))
    for x, z in ((2, 2), (6, 2)):
        out.append(Trajectory(
            'farming', f'Harvest the sugar cane at ({x}, 66, {z}) and leave the bottom block.',
            'Sugar cane regrows from the bottom; break only the upper blocks.',
            f"const cane = bot.findBlock({{ matching: b => b.name === 'sugar_cane', maxDistance: 8 }});\n"
            f"if (!cane) throw new Error('no sugar cane');\n"
            f'await bot.dig(cane);',
            'Harvested sugar cane above the base block.', 'I harvested sugar cane.',
            mock=mock(world={_key(x, 66, z): 'sugar_cane'}, inventory=['stone_hoe']),
            expect={'type': 'dig', 'block': 'sugar_cane'}))
    out.append(Trajectory(
        'farming', 'Check that the wheat farm is mature enough to harvest.',
        'Read the crop age instead of harvesting everything and wasting growth.',
        "const crops = Object.values(bot.registry.blocksByName).filter(b => b.name === 'wheat');\n"
        "const mature = bot.findBlock({ matching: b => b.name === 'wheat' && b.getProperties().age === 7, maxDistance: 32 });\n"
        "bot.chat(mature ? 'Wheat is ready to harvest' : 'Wheat is still growing');",
        'Checked crop maturity.', 'I checked the wheat.',
        mock=mock(world={_key(0, 64, 0): 'wheat'}, inventory=['wheat_seeds']),
        expect={'type': 'chat', 'contains': 'Wheat'}))
    for crop, seed, age in (('wheat', 'wheat_seeds', 7), ('carrots', 'carrot', 7),
                            ('potatoes', 'potatoes', 7), ('beetroot', 'beetroot_seeds', 3)):
        for x, z in ((12, 0), (16, 0)):
            out.append(Trajectory(
                'farming', f'Harvest the {crop} at ({x}, 64, {z}) and replant immediately.',
                'Replant in the same tick so the farm never loses a cycle.',
                f"const crop = bot.blockAt(new Vec3({x}, 64, {z}));\n"
                f"if (!crop || crop.name !== '{crop}') throw new Error('no {crop} here');\n"
                f'await bot.dig(crop);\n'
                f"const seeds = bot.inventory.findInventoryItem(bot.registry.itemsByName['{seed}'].id, null);\n"
                f"if (!seeds) throw new Error('no {seed}');\nawait bot.equip(seeds, 'hand');\n"
                f'await bot.placeBlock(bot.blockAt(new Vec3({x}, 63, {z})), new Vec3(0, 1, 0));',
                f'Harvested and replanted {crop}.', f'I kept the {crop} farm cycling.',
                mock=mock(world={_key(x, 64, z): crop, _key(x, 63, z): 'farmland'},
                          inventory=[seed]),
                expect={'type': 'place', 'block': seed}))
            out.append(Trajectory(
                'farming', f'Plant {seed.replace("_", " ")} in the prepared plot at ({x}, 64, {z}).',
                'Farmland below, air above; equip the seed and place once.',
                f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['{seed}'].id, null), 'hand');\n"
                f'await bot.placeBlock(bot.blockAt(new Vec3({x}, 63, {z})), new Vec3(0, 1, 0));',
                f'Planted {seed}.', f'I planted {seed.replace("_", " ")}.',
                mock=mock(world={_key(x, 63, z): 'farmland'}, inventory=[seed]),
                expect={'type': 'place', 'block': seed}))
            out.append(Trajectory(
                'farming', f'Bone-meal the {crop} at ({x}, 64, {z}) to speed up growth.',
                'Bone meal is limited; use it on the crop that is furthest behind.',
                f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['bone_meal'].id, null), 'hand');\n"
                f'await bot.activateBlock(bot.blockAt(new Vec3({x}, 64, {z})));',
                f'Applied bone meal to {crop}.', f'I fertilised the {crop}.',
                mock=mock(world={_key(x, 64, z): crop}, inventory=['bone_meal']),
                expect={'type': 'activateBlock', 'block': crop}))
    for x, z in ((20, 0), (24, 0), (28, 0)):
        out.append(Trajectory(
            'farming', f'Till the grass at ({x}, 64, {z}) for a new wheat plot.',
            'Expanding the farm needs farmland first; hoe the grass before planting.',
            f"await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['stone_hoe'].id, null), 'hand');\n"
            f'await bot.activateBlock(bot.blockAt(new Vec3({x}, 64, {z})));',
            'Grass converted to farmland.', 'I prepared a new plot.',
            mock=mock(world={_key(x, 64, z): 'grass_block'}, inventory=['stone_hoe']),
            expect={'type': 'activateBlock', 'block': 'grass_block'}))
    for x, z in ((4, 4), (8, 4), (12, 4)):
        out.append(Trajectory(
            'farming', f'Check the water level around the farm plot at ({x}, 64, {z}).',
            'Hydrated farmland grows faster; if it is dry, place water before planting.',
            f"const soil = bot.blockAt(new Vec3({x}, 63, {z}));\n"
            f"if (!soil || soil.getProperties().moisture < 7) bot.chat('Farmland is dry; bring water');\n"
            f"else bot.chat('Farmland is hydrated');",
            'Checked farmland moisture.', 'I checked the irrigation.',
            mock=mock(world={_key(x, 63, z): {'name': 'farmland', 'props': {'moisture': 3}}},
                      inventory=['water_bucket']),
            expect={'type': 'chat', 'contains': 'dry'}))
    return out


GENERATORS = {'navigation': _navigation, 'mining': _mining, 'crafting': _crafting,
              'combat': _combat, 'building': _building, 'inventory': _inventory,
              'chat': _chat, 'farming': _farming}


def mineflayer_trajectories(seed: int = 19) -> list[Trajectory]:
    """All trajectories, deterministically shuffled for a given seed."""
    rng = random.Random(seed)
    rows: list[Trajectory] = []
    for skill in SKILLS:
        rows.extend(GENERATORS[skill](rng))
    rng.shuffle(rows)
    _validate(rows)
    return rows


def mineflayer_records(seed: int = 19) -> list[str]:
    """Rendered UTF-8 training records for the ``mineflayer`` data group."""
    return [row.render() for row in mineflayer_trajectories(seed)]


def dataset_stats(seed: int = 19) -> dict:
    rows = mineflayer_trajectories(seed)
    counts = {skill: sum(r.skill == skill for r in rows) for skill in SKILLS}
    return {'total': len(rows), 'skills': counts,
            'bytes': sum(len(r.render().encode('utf8')) for r in rows),
            'mean_bytes': round(sum(len(r.render().encode('utf8')) for r in rows) / len(rows), 1),
            'multi_line_actions': sum('\n' in r.action for r in rows)}


def _validate(rows: list[Trajectory]) -> None:
    assert len(rows) >= 500, f'need 500+ trajectories, have {len(rows)}'
    for skill in SKILLS:
        assert sum(r.skill == skill for r in rows) >= 50, f'{skill} has too few rows'
    for row in rows:
        assert row.skill in SKILLS
        for value in (row.scenario, row.thought, row.observation, row.answer):
            assert value and '\n' not in value, f'one-line fields required: {row!r}'
        assert 'bot.' in row.action, f'missing bot call: {row.action!r}'
        assert row.expect.get('type'), f'missing expectation: {row.action!r}'


def write_jsonl(path, seed: int = 19) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as handle:
        for row in mineflayer_trajectories(seed):
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    return len(mineflayer_trajectories(seed))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write JSONL trajectories here')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--show', type=int, default=2, help='print N rendered examples')
    args = parser.parse_args(argv)
    rows = mineflayer_trajectories(args.seed)
    print(json.dumps(dataset_stats(args.seed), indent=2))
    for row in rows[:args.show]:
        print('\n' + '-' * 72 + '\n' + row.render())
    if args.output:
        print('wrote', write_jsonl(args.output, args.seed), 'rows to', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
