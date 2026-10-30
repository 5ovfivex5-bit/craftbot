// pybind11 module `_craftsim`: thin wrapper over craftsim::VecEnv.
// Output arrays are allocated here and filled in place by C++ (no extra copies).
// Indexing: agent i = world * K + k (N = W * K rows). Save/load, chests and tasks use the world index.
// Actions: action_mode "v7" -> int32[N, 4] (primary, recipe, item select, msg); "v6" -> int32[N].

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>

#include "gen/registry.h"
#include "sim.h"

namespace py = pybind11;
using namespace craftsim;
using namespace pybind11::literals;

using I32In = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using F32In = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

py::ssize_t S(long long v) { return (py::ssize_t)v; }

}  // namespace

PYBIND11_MODULE(_craftsim, m) {
    m.doc() = "Headless batched Minecraft-like voxel simulator (see src/sim.h, src/README_v7.md).";

    m.attr("NUM_ITEMS") = (int)NUM_ITEMS;
    m.attr("NUM_BLOCKS") = (int)NUM_BLOCKS;
    m.attr("NUM_RECIPES") = (int)NUM_RECIPES;
    m.attr("NUM_MOBS") = (int)NUM_MOBS;
    m.attr("NUM_TASKS") = (int)NUM_TASKS;
    m.attr("NUM_FAMILIES") = (int)NUM_FAMILIES;
    m.attr("NUM_PRIMARY") = (int)NUM_PRIMARY;
    m.attr("NUM_ACTION_COLS") = (int)NUM_ACTION_COLS;
    m.attr("NUM_MSGS") = (int)NUM_MSGS;
    m.attr("MAX_AGENTS") = (int)MAX_AGENTS;
    m.attr("V6_NUM_ITEMS") = (int)V6_NUM_ITEMS;
    m.attr("V6_NUM_BLOCKS") = (int)V6_NUM_BLOCKS;
    m.attr("V6_NUM_ACTIONS") = (int)V6_NUM_ACTIONS;
    m.attr("VEC_DIM") = (int)VEC7_DIM;
    m.attr("VEC_DIM_V6") = (int)VEC_DIM_V6;
    m.attr("WORLD_SIZE") = py::make_tuple(WX, WY, WZ);
    m.attr("SEA_LEVEL") = (int)SEA_LEVEL;
    m.attr("GOAL_OFFSET") = (int)GOAL_OFFSET;
    m.attr("V7_INV") = (int)V7_INV;
    m.attr("V7_SCALARS") = (int)V7_SCALARS;
    m.attr("V7_TASK") = (int)V7_TASK;
    m.attr("V7_TASK_PARAM") = (int)V7_TASK_PARAM;
    m.attr("V7_TEAM") = (int)V7_TEAM;
    m.attr("V7_CHEST") = (int)V7_CHEST;
    m.attr("V7_XHAIR") = (int)V7_XHAIR;
    m.attr("V7_MOBS") = (int)V7_MOBS;
    m.attr("V7_BUILD") = (int)V7_BUILD;
    m.attr("V7_NUM_SCALARS") = (int)V7_NUM_SCALARS;

    // Registry (generated from data/ by tools/gen_registry.py): names and key fields, index == id.
    m.def("registry", []() {
        py::dict d;
        py::list blocks, items, recipes, mobs, tasks, families;
        for (int i = 0; i < NUM_BLOCKS; i++) {
            const reg::BlockDef& b = reg::BLOCKS[i];
            blocks.append(py::dict("name"_a = b.name, "hardness"_a = b.hardness, "tool"_a = (int)b.bestTool,
                                   "min_tier"_a = (int)b.minTier, "solid"_a = (bool)b.solid,
                                   "transparent"_a = (bool)b.transparent, "liquid"_a = (bool)b.liquid,
                                   "pseudo"_a = (bool)b.pseudo, "drop"_a = (int)b.drop));
        }
        for (int i = 0; i < NUM_ITEMS; i++) {
            const reg::ItemDef& it = reg::ITEMS[i];
            items.append(py::dict("name"_a = it.name, "stack"_a = (int)it.stack, "kind"_a = (int)it.kind,
                                  "places"_a = (int)it.places, "tool_type"_a = (int)it.toolType,
                                  "tool_tier"_a = (int)it.toolTier, "fuel_ticks"_a = (int)it.fuelTicks,
                                  "value"_a = (int)it.value));
        }
        for (int i = 0; i < NUM_RECIPES; i++) {
            const reg::RecipeDef& r = reg::RECIPES[i];
            py::dict ing;
            for (int j = 0; j < r.nIng; j++) ing[py::int_(r.ingItem[j])] = (int)r.ingCount[j];
            recipes.append(py::dict("name"_a = r.name, "result"_a = (int)r.result, "count"_a = (int)r.count,
                                    "needs"_a = (int)r.needs, "cook_ticks"_a = (int)r.cookTicks,
                                    "ingredients"_a = ing));
        }
        for (int i = 0; i < NUM_MOBS; i++) {
            const reg::MobDef& mo = reg::MOBS[i];
            mobs.append(py::dict("name"_a = mo.name, "block"_a = (int)mo.block, "hostile"_a = (bool)mo.hostile,
                                 "health"_a = mo.health, "damage"_a = mo.damage));
        }
        for (int i = 0; i < NUM_TASKS; i++) {
            const reg::TaskDef& t = reg::TASKS[i];
            tasks.append(py::dict("name"_a = t.name, "family"_a = (int)t.family, "max_steps"_a = (int)t.maxSteps,
                                  "success_type"_a = (int)t.success.type, "success_a"_a = (int)t.success.a,
                                  "success_b"_a = (int)t.success.b, "milestones"_a = (int)t.nMilestones));
        }
        for (int i = 0; i < NUM_FAMILIES; i++) families.append(reg::FAMILY_NAMES[i]);
        d["blocks"] = blocks;
        d["items"] = items;
        d["recipes"] = recipes;
        d["mobs"] = mobs;
        d["tasks"] = tasks;
        d["families"] = families;
        return d;
    });

    py::class_<Snapshot>(m, "Snapshot")
        .def_readonly("x", &Snapshot::x)
        .def_readonly("y", &Snapshot::y)
        .def_readonly("z", &Snapshot::z)
        .def_readonly("vx", &Snapshot::vx)
        .def_readonly("vy", &Snapshot::vy)
        .def_readonly("vz", &Snapshot::vz)
        .def_readonly("yaw", &Snapshot::yaw)
        .def_readonly("pitch", &Snapshot::pitch)
        .def_readonly("on_ground", &Snapshot::onGround)
        .def_readonly("health", &Snapshot::health)
        .def_readonly("t", &Snapshot::t)
        .def_readonly("ticks", &Snapshot::ticks)
        .def_readonly("seed", &Snapshot::seed)
        .def_readonly("episode", &Snapshot::episode)
        .def_readonly("last_action", &Snapshot::lastAction)
        .def_readonly("msg", &Snapshot::msg)
        .def_readonly("mining_progress", &Snapshot::miningProgress)
        .def_readonly("goal", &Snapshot::goal)
        .def_readonly("kit", &Snapshot::kit)
        .def_readonly("restart", &Snapshot::restart)
        .def_readonly("has_target", &Snapshot::hasTarget)
        .def_readonly("tx", &Snapshot::tx)
        .def_readonly("ty", &Snapshot::ty)
        .def_readonly("tz", &Snapshot::tz)
        .def_readonly("tblock", &Snapshot::tblock)
        .def_readonly("held", &Snapshot::held)
        .def_readonly("dim", &Snapshot::dim)
        .def_readonly("task", &Snapshot::task)
        .def_readonly("task_max_t", &Snapshot::taskMaxT)
        .def_readonly("sneak", &Snapshot::sneak);

    py::class_<VecEnv>(m, "VecEnv")
        .def(py::init<int, int, uint64_t, int, int, int, int, int>(),
             py::arg("num_envs"), py::arg("agents"), py::arg("seed"), py::arg("res"),
             py::arg("max_steps"), py::arg("ticks"), py::arg("threads"), py::arg("mode") = (int)MODE_V7)
        .def_property_readonly("num_envs", &VecEnv::numWorlds)
        .def_property_readonly("num_agents", &VecEnv::size)
        .def_property_readonly("agents", &VecEnv::agentsPerWorld)
        .def_property_readonly("res", &VecEnv::res)
        .def_property_readonly("mode", &VecEnv::mode)
        .def_property_readonly("vec_dim", &VecEnv::vecDim)
        .def_property_readonly("action_cols", &VecEnv::actionCols)
        .def("reset", [](VecEnv& s) {
            const int n = s.size(), R = s.res();
            py::array_t<uint8_t> img({S(n), S(R), S(R), S(2)});
            py::array_t<float> vec({S(n), S(s.vecDim())});
            {
                py::gil_scoped_release g;
                s.reset(img.mutable_data(), vec.mutable_data());
            }
            return py::make_tuple(img, vec);
        })
        .def("step", [](VecEnv& s, const I32In& act, const py::object& msg) {
            const int n = s.size(), R = s.res();
            if (act.size() != (py::ssize_t)n * s.actionCols())
                throw std::invalid_argument("actions must have num_agents x action_cols values");
            I32In m;
            const int32_t* mp = nullptr;
            if (!msg.is_none()) {
                m = msg.cast<I32In>();
                if (m.size() != n) throw std::invalid_argument("msgs must have length num_agents");
                mp = m.data();
            }
            py::array_t<uint8_t> img({S(n), S(R), S(R), S(2)});
            py::array_t<float> vec({S(n), S(s.vecDim())});
            py::array_t<float> rew(S(n));
            py::array_t<bool> done(S(n));
            py::array_t<uint32_t> ach(S(n));
            {
                py::gil_scoped_release g;
                s.step(act.data(), mp, img.mutable_data(), vec.mutable_data(), rew.mutable_data(),
                       done.mutable_data(), ach.mutable_data());
            }
            return py::make_tuple(img, vec, rew, done, ach);
        }, py::arg("actions"), py::arg("msgs"))
        .def("last_msgs", [](const VecEnv& s) {
            py::array_t<int32_t> out(S(s.size()));
            s.lastMsgs(out.mutable_data());
            return out;
        })
        .def("reset_env", [](VecEnv& s, int i, uint64_t seed) {
            // Resets the world of agent i; returns the obs rows of that world's K agents.
            const int K = s.agentsPerWorld(), R = s.res();
            const int w = i / K;
            if (i < 0 || i >= s.size()) throw std::out_of_range("agent index out of range");
            py::array_t<uint8_t> img({S(K), S(R), S(R), S(2)});
            py::array_t<float> vec({S(K), S(s.vecDim())});
            s.resetWorld(w, seed, img.mutable_data(), vec.mutable_data());
            return py::make_tuple(img, vec);
        }, py::arg("i"), py::arg("seed"))
        .def("set_start_kits", [](VecEnv& s, F32In table, F32In prob) {
            if (table.ndim() != 2 || table.shape(0) != NUM_ITEMS || table.shape(1) != NUM_ITEMS)
                throw std::invalid_argument("start kit table must be [NUM_ITEMS, NUM_ITEMS]");
            if (prob.ndim() != 1 || prob.shape(0) != NUM_ITEMS)
                throw std::invalid_argument("start kit prob must be [NUM_ITEMS]");
            s.setStartKits(table.data(), prob.data());
        }, py::arg("table"), py::arg("prob"))
        .def("set_novelty", &VecEnv::setNovelty, py::arg("scale"))
        .def_property_readonly("novelty", &VecEnv::novelty)
        .def("set_team_share", &VecEnv::setTeamShare, py::arg("share"))
        .def_property_readonly("team_share", &VecEnv::teamShare)
        .def("last_novelty", [](const VecEnv& s) {
            py::array_t<float> out(S(s.size()));
            s.lastNovelty(out.mutable_data());
            return out;
        })
        .def("save_state", [](const VecEnv& s, int w) {
            std::vector<uint8_t> b = s.saveState(w);
            return py::bytes(reinterpret_cast<const char*>(b.data()), b.size());
        }, py::arg("w"))
        .def("load_state", [](VecEnv& s, int w, const std::string& blob, int goal) {
            const int K = s.agentsPerWorld(), R = s.res();
            py::array_t<uint8_t> img({S(K), S(R), S(R), S(2)});
            py::array_t<float> vec({S(K), S(s.vecDim())});
            s.loadState(w, reinterpret_cast<const uint8_t*>(blob.data()), blob.size(), goal,
                        img.mutable_data(), vec.mutable_data());
            return py::make_tuple(img, vec);
        }, py::arg("w"), py::arg("blob"), py::arg("goal"))
        .def("chests", [](const VecEnv& s, int w) {
            py::list out;
            for (const ChestView& c : s.chests(w)) {
                py::array_t<int32_t> counts(S(NUM_ITEMS));
                std::memcpy(counts.mutable_data(), c.count, sizeof(c.count));
                out.append(py::make_tuple(c.x, c.y, c.z, counts));
            }
            return out;
        }, py::arg("w"))
        .def("render", [](const VecEnv& s, int i, int W, int H) {
            if (W < 1 || H < 1) throw std::invalid_argument("width/height must be >= 1");
            py::array_t<uint8_t> out({S(H), S(W), S(3)});
            s.render(i, W, H, out.mutable_data());
            return out;
        }, py::arg("i"), py::arg("width"), py::arg("height"))
        .def("inventory", [](const VecEnv& s, int i) {
            py::array_t<int32_t> out(S(NUM_ITEMS));
            s.inventory(i, out.mutable_data());
            return out;
        }, py::arg("i"))
        .def("set_inventory", &VecEnv::setInventory, py::arg("i"), py::arg("item"), py::arg("count"))
        .def("set_goal_weights", [](VecEnv& s, F32In w) {
            if (w.size() != NUM_ITEMS) throw std::invalid_argument("goal weights must have NUM_ITEMS entries");
            s.setGoalWeights(w.data());
        }, py::arg("weights"))
        .def("goal", &VecEnv::goal, py::arg("i"))
        .def("set_goal", &VecEnv::setGoal, py::arg("i"), py::arg("item"))
        .def("set_task_weights", [](VecEnv& s, int w, F32In weights) {
            if (weights.size() != NUM_TASKS) throw std::invalid_argument("task weights must have NUM_TASKS entries");
            s.setTaskWeights(w, weights.data());
        }, py::arg("w"), py::arg("weights"))
        .def("set_task", &VecEnv::setTask, py::arg("w"), py::arg("task"), py::arg("p0"), py::arg("p1"))
        .def("task", &VecEnv::task, py::arg("i"))
        .def("set_shaping_table", [](VecEnv& s, int task, I32In items) {
            s.setShapingTable(task, items.data(), (int)items.size());
        }, py::arg("task"), py::arg("items"))
        .def("time_of_day", &VecEnv::timeOfDay, py::arg("w"))
        .def("world_seed", &VecEnv::worldSeed, py::arg("w"))
        .def("build_target", [](const VecEnv& s, int w) {
            const auto v = s.buildTarget(w);
            py::dict d;
            d["blueprint"] = v.blueprint;
            d["rot"] = v.rot;
            py::list cells;
            for (const auto& c : v.cells) cells.append(py::make_tuple(c[0], c[1], c[2], c[3], c[4]));
            d["cells"] = cells;
            return d;
        }, py::arg("w"))
        .def("entities", [](const VecEnv& s, int w) {
            py::list out;
            for (const auto& e : s.entities(w)) {
                py::dict d;
                d["type"] = std::string(reg::MOBS[e.type].name);
                d["x"] = e.x; d["y"] = e.y; d["z"] = e.z;
                d["health"] = e.hp;
                out.append(d);
            }
            return out;
        }, py::arg("w"))
        .def("state", &VecEnv::state, py::arg("i"))
        .def("set_pose", &VecEnv::setPose, py::arg("i"), py::arg("yaw"), py::arg("pitch"))
        .def("block", &VecEnv::block, py::arg("i"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("region", [](const VecEnv& s, int i, int radius) {
            Region r = s.region(i, radius);
            py::array_t<uint8_t> out({S(r.sy), S(r.sz), S(r.sx)});
            if (!r.blocks.empty()) std::memcpy(out.mutable_data(), r.blocks.data(), r.blocks.size());
            return py::make_tuple(py::make_tuple(r.ox, r.oy, r.oz), out);
        }, py::arg("i"), py::arg("radius"));
}
