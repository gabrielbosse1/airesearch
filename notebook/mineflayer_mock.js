'use strict';
/*
 * Phase 19.4 — mock Mineflayer harness.
 *
 * Reads {"cases": [{code, mock}]} on stdin and writes
 * {"results": [{ok, error, actions, position, inventory, logs}]} on stdout.
 *
 * This is a *test double* of the Mineflayer API. It never opens a socket, never
 * touches a real Minecraft server and never requires the mineflayer package.
 * Generated code runs inside node:vm with string code generation disabled and
 * without require/process/fs, with a wall-clock race. That is isolation for
 * non-adversarial model output, NOT a security boundary: do not feed it code
 * from an untrusted third party.
 */
const vm = require('node:vm');

const CASE_TIMEOUT_MS = 3000;

class Vec3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  plus(o) { return new Vec3(this.x + o.x, this.y + o.y, this.z + o.z); }
  minus(o) { return new Vec3(this.x - o.x, this.y - o.y, this.z - o.z); }
  offset(dx, dy, dz) { return new Vec3(this.x + dx, this.y + dy, this.z + dz); }
  distanceTo(o) { return Math.hypot(this.x - o.x, this.y - o.y, this.z - o.z); }
  clone() { return new Vec3(this.x, this.y, this.z); }
  floored() { return new Vec3(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z)); }
  toString() { return `(${this.x}, ${this.y}, ${this.z})`; }
}

class Goal {
  constructor() { this.kind = this.constructor.name; }
  target() { return null; }
}
class GoalNear extends Goal {
  constructor(x, y, z, range = 1) { super(); this.x = x; this.y = y; this.z = z; this.range = range; }
  target() { return [this.x, this.y, this.z]; }
}
class GoalBlock extends Goal {
  constructor(x, y, z) { super(); this.x = x; this.y = y; this.z = z; this.range = 0; }
  target() { return [this.x, this.y, this.z]; }
}
class GoalXZ extends Goal {
  constructor(x, z) { super(); this.x = x; this.z = z; this.y = null; this.range = 0; }
  target() { return [this.x, null, this.z]; }
}
class GoalFollow extends Goal {
  constructor(entity, range = 2) {
    super(); this.entity = entity; this.range = range;
    this.entityName = entity && entity.name ? entity.name : null;
  }
  target() {
    const p = this.entity && this.entity.position;
    return p ? [p.x, p.y, p.z] : null;
  }
}

const key = (x, y, z) => `${Math.floor(x)},${Math.floor(y)},${Math.floor(z)}`;

// Mature crops unless the case supplies explicit properties.
const DEFAULT_PROPS = {
  wheat: { age: 7 }, carrots: { age: 7 }, potatoes: { age: 7 }, beetroot: { age: 3 },
  farmland: { moisture: 7 }
};

function makeItem(name, count, id) {
  return { name, count: count || 1, type: id || 0, slot: 0, metadata: null };
}

function buildCase(spec) {
  spec = spec || {};
  const actions = [];
  const logs = [];
  const world = new Map();
  for (const [k, value] of Object.entries(spec.world || {})) {
    const entry = typeof value === 'string' ? { name: value, props: {} } : { name: value.name, props: value.props || {} };
    entry.props = Object.assign({}, DEFAULT_PROPS[entry.name] || {}, entry.props);
    world.set(k, entry);
  }
  const registry = spec.registry || {};
  const idToName = new Map();
  const itemsByName = {};
  const blocksByName = {};
  for (const [name, id] of Object.entries(registry)) {
    itemsByName[name] = { id, name };
    blocksByName[name] = { id, name, displayName: name };
    idToName.set(id, name);
  }
  const inventory = (spec.inventory || []).map(row => {
    const name = typeof row === 'string' ? row : row.name;
    const count = typeof row === 'string' ? 1 : (row.count || 1);
    return makeItem(name, count, registry[name] || 0);
  });
  const playerPos = new Vec3(...(spec.position || [0, 64, 0]));
  const entityList = (spec.entities || []).map(e => ({
    id: e.id || 1, name: e.name, type: e.type || 'mob', health: e.health || 20,
    isValid: true, position: new Vec3(...(e.position || [0, 64, 0]))
  }));
  const entities = {};
  for (const e of entityList) entities[e.id] = e;
  const players = {};
  for (const e of entityList) {
    if (e.type === 'player') players[e.name] = { username: e.name, entity: e };
  }
  let held = null;

  const itemOf = itemType => {
    if (itemType && typeof itemType === 'object') return itemType;
    const name = idToName.get(itemType);
    return inventory.find(i => i.name === name) || null;
  };

  const blockAt = point => {
    if (point == null || point.x === undefined) throw new Error('blockAt expects a Vec3');
    const entry = world.get(key(point.x, point.y, point.z));
    const name = entry ? entry.name : 'air';
    const props = entry ? entry.props : {};
    return {
      name, position: new Vec3(Math.floor(point.x), Math.floor(point.y), Math.floor(point.z)),
      metadata: props.age || 0, getProperties: () => ({ ...props }), boundingBox: 'block',
      diggable: name !== 'air'
    };
  };

  const findBlock = options => {
    if (!options || typeof options.matching !== 'function') throw new Error('findBlock expects {matching}');
    const max = options.maxDistance === undefined ? 16 : options.maxDistance;
    for (const [k, entry] of world.entries()) {
      const [x, y, z] = k.split(',').map(Number);
      const block = blockAt(new Vec3(x, y, z));
      const distance = block.position.distanceTo(playerPos);
      if (distance <= max && options.matching(block)) return block;
    }
    return null;
  };

  const recipesFor = (itemType, metadata, minResultCount, craftingTable) => {
    const name = nameOf(itemType);
    if (!name) return [];
    if (TABLE_RECIPES.has(name) && !craftingTable) return [];
    return [{ result: { id: itemType, name, count: 1 }, requiresTable: TABLE_RECIPES.has(name) }];
  };
  const nameOf = itemType => {
    if (typeof itemType === 'string') return itemType;
    return idToName.get(itemType) || (itemsByName[itemType] && itemsByName[itemType].name) || null;
  };

  const record = (type, data) => actions.push(Object.assign({ type }, data || {}));
  const assertBlock = (block, verb) => {
    if (!block) throw new Error(`${verb}: block is null (not loaded or wrong coordinates)`);
    if (block.name === 'air') throw new Error(`${verb}: block at ${block.position} is air`);
    return block;
  };
  const performDig = (block, verb) => {
    assertBlock(block, verb);
    if (!block.diggable) throw new Error(`${verb}: block is not diggable`);
    record('dig', { block: block.name, position: [block.position.x, block.position.y, block.position.z] });
    world.set(key(block.position.x, block.position.y, block.position.z), { name: 'air', props: {} });
    return Promise.resolve();
  };

  const bot = {
    username: 'ByteCoreBot', version: '1.20.1',
    entity: { position: playerPos, height: 1.62, onGround: true, velocity: new Vec3(0, 0, 0) },
    entities, players,
    health: spec.health === undefined ? 20 : spec.health,
    food: spec.food === undefined ? 20 : spec.food,
    experience: { level: 0, points: 0 },
    time: { timeOfDay: 6000, day: 1, isDay: true },
    game: { gameMode: 'survival', dimension: 'overworld', difficulty: 'normal' },
    registry: { itemsByName, blocksByName },
    get inventory() {
      return {
        items: () => inventory.map(i => ({ ...i })),
        findInventoryItem: itemType => {
          const name = typeof itemType === 'number' ? idToName.get(itemType) : itemType && itemType.name;
          return inventory.find(i => i.name === name) || null;
        },
        emptySlotCount: () => Math.max(0, 36 - inventory.length),
        slots: inventory
      };
    },
    blockAt, findBlock, recipesFor,
    recipesAll: (itemType, metadata, craftingTable) => {
      const name = nameOf(itemType);
      if (!name) return [];
      if (TABLE_RECIPES.has(name) && !craftingTable) return [];
      return [{ result: { id: itemType, name, count: 1 }, requiresTable: TABLE_RECIPES.has(name) }];
    },
    dig: (block, forceLook, digFace) => performDig(block, 'bot.dig'),
    stopDigging: () => Promise.resolve(),
    digTime: () => 750,
    placeBlock: (reference, face) => {
      if (!reference) throw new Error('bot.placeBlock: reference block is null');
      if (reference.name === 'air') throw new Error(`bot.placeBlock: cannot place against air at ${reference.position}`);
      if (!face || face.x === undefined) throw new Error('bot.placeBlock: face must be a Vec3');
      if (!held) throw new Error('bot.placeBlock: nothing is equipped');
      const p = reference.position.plus(face);
      record('placeBlock', { block: held.name, position: [p.x, p.y, p.z], against: reference.name });
      world.set(key(p.x, p.y, p.z), { name: held.name, props: {} });
      return Promise.resolve();
    },
    equip: (item, destination) => {
      if (!item) throw new Error('bot.equip: item is null (missing from inventory)');
      record('equip', { item: item.name, destination });
      if (destination === 'hand') held = item;
      return Promise.resolve();
    },
    unequip: destination => { record('unequip', { destination }); return Promise.resolve(); },
    toss: (itemType, metadata, count) => {
      const item = itemOf(itemType);
      if (!item) throw new Error('bot.toss: item not in inventory');
      record('toss', { item: item.name, count: count === undefined || count === null ? item.count : count });
      return Promise.resolve();
    },
    tossStack: item => {
      if (!item) throw new Error('bot.tossStack: item is null');
      record('toss', { item: item.name, count: item.count });
      return Promise.resolve();
    },
    consume: () => {
      if (!held) throw new Error('bot.consume: no item in hand');
      record('consume', { item: held.name });
      return Promise.resolve();
    },
    craft: (recipe, count, craftingTable) => {
      if (!recipe || !recipe.result) throw new Error('bot.craft: invalid recipe (check recipesFor result)');
      if (recipe.requiresTable && !craftingTable) throw new Error('bot.craft: recipe requires a crafting table');
      const amount = count === undefined || count === null ? 1 : count;
      record('craft', { item: recipe.result.name, count: amount, table: !!craftingTable });
      const existing = inventory.find(i => i.name === recipe.result.name);
      if (existing) existing.count += amount * (recipe.result.count || 1);
      else inventory.push(makeItem(recipe.result.name, amount * (recipe.result.count || 1), registry[recipe.result.name] || 0));
      return Promise.resolve();
    },
    attack: entity => {
      if (!entity) throw new Error('bot.attack: target entity is null');
      record('attack', { entity: entity.name || 'unknown' });
      return Promise.resolve();
    },
    chat: message => {
      if (typeof message !== 'string') throw new Error('bot.chat expects a string');
      record('chat', { message }); logs.push(message); return message;
    },
    whisper: (username, message) => {
      if (typeof username !== 'string' || typeof message !== 'string') throw new Error('bot.whisper expects strings');
      record('whisper', { player: username, message }); return message;
    },
    nearestEntity: filter => {
      const candidates = entityList.filter(e => e.isValid && (typeof filter === 'function' ? filter(e) : true));
      candidates.sort((a, b) => a.position.distanceTo(playerPos) - b.position.distanceTo(playerPos));
      return candidates[0] || null;
    },
    openContainer: block => {
      if (!block) throw new Error('bot.openContainer: block is null');
      record('openContainer', { block: block.name });
      return Promise.resolve({
        deposit: (itemType, metadata, count) => {
          const item = itemOf(itemType);
          if (!item) throw new Error('deposit: item not in inventory');
          record('deposit', { item: item.name, count });
          return Promise.resolve();
        },
        withdraw: (itemType, metadata, count) => {
          record('withdraw', { item: nameOf(itemType), count });
          return Promise.resolve();
        },
        close: () => { record('closeContainer', {}); },
        containerItems: () => []
      });
    },
    activateBlock: block => {
      if (!block) throw new Error('bot.activateBlock: block is null');
      record('activateBlock', { block: block.name });
      return Promise.resolve();
    },
    activateItem: () => { record('activateItem', {}); return Promise.resolve(); },
    deactivateItem: () => Promise.resolve(),
    lookAt: () => Promise.resolve(),
    look: () => Promise.resolve(),
    setControlState: () => {},
    clearControlStates: () => {},
    setQuickBarSlot: () => {},
    waitForTicks: () => Promise.resolve(),
    loadPlugin: () => bot,
    on: (event, handler) => { record('listen', { event }); return bot; },
    once: (event, handler) => { record('listen', { event }); return bot; },
    removeListener: () => bot,
    removeAllListeners: () => bot,
    emit: () => false,
    quit: () => { record('quit', {}); },
    pathfinder: {
      goto: goal => {
        if (!goal || typeof goal.target !== 'function') throw new Error('pathfinder.goto expects a Goal instance');
        const target = goal.target();
        if (!target && !goal.entityName) throw new Error('pathfinder.goto: goal has no destination');
        record('goto', {
          goal: goal.kind, target: goal.entityName ? null : target, within: goal.range,
          follow: goal.entityName || null
        });
        if (target) playerPos.x = target[0], playerPos.y = target[1] === null ? playerPos.y : target[1], playerPos.z = target[2];
        return Promise.resolve();
      },
      setGoal: goal => {
        if (!goal || typeof goal.target !== 'function') throw new Error('pathfinder.setGoal expects a Goal instance');
        record('setGoal', { goal: goal.kind, target: goal.entityName ? null : goal.target(), within: goal.range, follow: goal.entityName || null });
      },
      stop: () => { record('stop', {}); },
      setMovements: () => {},
      isMoving: () => false
    },
    pvp: {
      attack: entity => {
        if (!entity) throw new Error('bot.pvp.attack: entity is null');
        record('pvp_attack', { entity: entity.name || (entity.entity && entity.entity.name) || 'unknown' });
      },
      stop: () => {}
    },
    collectBlock: {
      collect: block => performDig(block, 'bot.collectBlock.collect')
    }
  };

  const sandbox = {
    bot, Vec3, GoalNear, GoalBlock, GoalXZ, GoalFollow,
    console: {
      log: (...args) => logs.push(args.map(String).join(' ')),
      info: (...args) => logs.push(args.map(String).join(' ')),
      warn: (...args) => logs.push(args.map(String).join(' ')),
      error: (...args) => logs.push(args.map(String).join(' '))
    },
    Math, JSON, Object, Array, Number, String, Boolean, Promise, Date, Error, isNaN, parseInt, parseFloat
  };
  return { sandbox, actions, logs, bot };
}

const TABLE_RECIPES = new Set([
  'furnace', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe',
  'wooden_sword', 'stone_sword', 'iron_sword', 'diamond_sword', 'bread', 'ladder',
  'bucket', 'chest', 'iron_helmet', 'iron_chestplate', 'iron_leggings', 'iron_boots',
  'wooden_hoe', 'stone_hoe', 'iron_hoe', 'shield', 'bow', 'stone_axe', 'iron_axe'
]);

async function runCase(spec) {
  const { sandbox, actions, logs, bot } = buildCase(spec.mock || {});
  let error = null;
  try {
    const context = vm.createContext(sandbox, { codeGeneration: { strings: false, wasm: false } });
    const script = new vm.Script(`(async () => {\n${spec.code}\n})()`, { filename: 'mineflayer_player.js' });
    const execution = script.runInContext(context, { timeout: CASE_TIMEOUT_MS });
    let timer;
    const timeout = new Promise(resolve => { timer = setTimeout(() => resolve('__CASE_TIMEOUT__'), CASE_TIMEOUT_MS); });
    const outcome = await Promise.race([Promise.resolve(execution), timeout]);
    clearTimeout(timer);
    if (outcome === '__CASE_TIMEOUT__') error = 'timeout: code did not settle within ' + CASE_TIMEOUT_MS + 'ms';
  } catch (exc) {
    error = `${exc && exc.name ? exc.name : 'Error'}: ${exc && exc.message ? exc.message : String(exc)}`;
  }
  return {
    ok: error === null, error, actions, logs,
    position: [bot.entity.position.x, bot.entity.position.y, bot.entity.position.z],
    health: bot.health, inventory: bot.inventory.items().map(i => ({ name: i.name, count: i.count }))
  };
}

async function main() {
  let raw = '';
  process.stdin.setEncoding('utf8');
  for await (const chunk of process.stdin) raw += chunk;
  let payload;
  try {
    payload = JSON.parse(raw || '{}');
  } catch (exc) {
    process.stdout.write(JSON.stringify({ results: [], fatal: 'invalid JSON: ' + exc.message }));
    return;
  }
  const results = [];
  for (const [index, spec] of (payload.cases || []).entries()) {
    try {
      const result = await runCase(spec);
      result.id = spec.id === undefined ? index : spec.id;
      results.push(result);
    } catch (exc) {
      results.push({ id: spec.id === undefined ? index : spec.id, ok: false,
        error: 'harness failure: ' + exc.message, actions: [], logs: [] });
    }
  }
  process.stdout.write(JSON.stringify({ results }));
}

main().catch(exc => {
  process.stdout.write(JSON.stringify({ results: [], fatal: String(exc && exc.stack || exc) }));
  process.exitCode = 1;
});
