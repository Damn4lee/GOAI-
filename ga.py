# -*- coding: utf-8 -*-
"""材料构效关系遗传算法（修复版）。

相对 Notebook 内旧版 GA 的改进：
- 随机种子可复现（GA_SEED 环境变量或命令行参数）；
- 适应度按基因哈希缓存，重复方案不重复调 LLM；
- 每个个体多次采样取中位数，降低 LLM 打分噪声。

用法：
    python ga.py "LiFePO4 cathode doping rate performance"
"""

import copy
import hashlib
import json
import os
import random
import statistics
import sys
import time

from config import CONFIG, llm_client, llm_extra_params, state

MAX_GENERATIONS = 5
POPULATION_SIZE = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3
EVAL_SAMPLES = 2

DEFAULT_SEARCH_SPACE = {
    "base_material": ["LiFePO4", "Na3V2(PO4)3"],
    "doping_element": ["Mg", "Al", "Ti", "None"],
    "doping_ratio": [0.01, 0.03, 0.05, 0.07],
    "coating_layer": ["Carbon", "Al2O3", "None"],
    "synthesis_temp_c": [600, 650, 700, 750],
    "calcination_time_h": [4, 6, 8, 10],
}

SEARCH_SPACE = CONFIG.get("ga_search_space", DEFAULT_SEARCH_SPACE)


def parse_llm_json(text: str):
    """从 LLM 输出中提取 JSON，兼容 markdown 代码块与截断修复。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end <= start:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"LLM 输出无法解析为 JSON: {cleaned[:200]}")


class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0.0
        self.eval_reason = ""

    def to_dict(self):
        return {"genes": self.genes, "fitness": self.fitness, "eval_reason": self.eval_reason}


class MaterialGA:
    def __init__(self, query=None, seed=None):
        if seed is None:
            seed = int(os.environ.get("GA_SEED", "42"))
        random.seed(seed)
        self.seed = seed
        self.query = query or CONFIG.get(
            "ga_default_query",
            "LiFePO4 cathode doping rate performance material structure property relationship",
        )
        self.population = []
        self.generation = 0
        self.history = []
        self._fitness_cache = {}
        self.literature_evidence = []

    def _log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GA] {msg}"
        print(line)
        try:
            with open("ga_log.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _call_llm(self, system_prompt, user_prompt):
        resp = llm_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
            extra_body=llm_extra_params(),
        )
        return resp.choices[0].message.content

    def initialize_population(self):
        from literature_searcher import sciverse_client

        self._log(f"检索文献: {self.query}")
        try:
            res = sciverse_client.search(self.query, page_size=10)
            self.literature_evidence = [{"gap": "default", "results": res}]
            lit = json.dumps(res, ensure_ascii=False, indent=2)[:6000]
        except Exception as e:  # noqa: BLE001
            self._log(f"检索失败({e})，使用领域通识模式")
            lit = "（文献检索不可用，请基于材料科学通识生成方案）"

        system = (
            "你是材料科学专家。根据文献检索结果设计候选材料合成方案。\n"
            f"参数与合法取值: {json.dumps(SEARCH_SPACE, ensure_ascii=False)}\n"
            "每个参数的值必须从合法取值中选取。严格返回 JSON 数组。"
        )
        user = f"文献检索结果:\n{lit}\n\n请生成 {POPULATION_SIZE} 个候选方案。"
        candidates = parse_llm_json(self._call_llm(system, user))

        self.population = []
        for c in candidates[:POPULATION_SIZE]:
            genes = {}
            for k, legal in SEARCH_SPACE.items():
                genes[k] = c.get(k) if c.get(k) in legal else random.choice(legal)
            self.population.append(Individual(genes))
        while len(self.population) < POPULATION_SIZE:
            self.population.append(self._mutate(copy.deepcopy(random.choice(self.population))))
        self._log(f"初始种群: {len(self.population)} 个个体")

    @staticmethod
    def _fitness_key(genes):
        return hashlib.md5(json.dumps(genes, sort_keys=True).encode("utf-8")).hexdigest()

    def _score_once(self, genes):
        system = (
            "你是严格的科学证据评审专家。对给定方案按证据、机制、相关性、新颖性综合评分(0-10)。\n"
            '严格返回 JSON: {"score": <float>, "reason": "<简要理由>"}'
        )
        user = "方案:\n" + json.dumps(genes, ensure_ascii=False, indent=2)
        r = parse_llm_json(self._call_llm(system, user))
        return max(0.0, min(10.0, float(r.get("score", 0)))), str(r.get("reason", ""))

    def evaluate_fitness(self):
        for ind in self.population:
            key = self._fitness_key(ind.genes)
            if key in self._fitness_cache:  # 适应度缓存：重复方案不重复花钱
                ind.fitness, ind.eval_reason = self._fitness_cache[key]
                continue
            scores, reasons = [], []
            for _ in range(EVAL_SAMPLES):  # 多采样取中位数降噪
                try:
                    s, r = self._score_once(ind.genes)
                    scores.append(s)
                    reasons.append(r)
                except Exception as e:  # noqa: BLE001
                    self._log(f"评估异常: {e}")
            if not scores:
                ind.fitness, ind.eval_reason = 0.0, "评估失败"
            else:
                ind.fitness = statistics.median(scores)
                ind.eval_reason = "; ".join(reasons[:2])
            self._fitness_cache[key] = (ind.fitness, ind.eval_reason)

    def _roulette_select(self):
        total = sum(i.fitness for i in self.population)
        if total <= 0:
            return random.choice(self.population)
        r = random.uniform(0, total)
        cum = 0.0
        for ind in self.population:
            cum += ind.fitness
            if cum >= r:
                return ind
        return self.population[-1]

    def _crossover(self, p1, p2):
        if random.random() > CROSSOVER_RATE:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        keys = list(SEARCH_SPACE.keys())
        pt = random.randint(1, len(keys) - 1)
        c1, c2 = {}, {}
        for idx, k in enumerate(keys):
            if idx < pt:
                c1[k], c2[k] = p1.genes[k], p2.genes[k]
            else:
                c1[k], c2[k] = p2.genes[k], p1.genes[k]
        return Individual(c1), Individual(c2)

    def _mutate(self, ind):
        for k in SEARCH_SPACE:
            if random.random() < MUTATION_RATE:
                ind.genes[k] = random.choice(SEARCH_SPACE[k])
        return ind

    def evolve_one_generation(self):
        new_pop = []
        best = max(self.population, key=lambda x: x.fitness)
        elite = copy.deepcopy(best)
        elite.eval_reason = "[精英保留] " + elite.eval_reason
        new_pop.append(elite)
        while len(new_pop) < POPULATION_SIZE:
            c1, c2 = self._crossover(self._roulette_select(), self._roulette_select())
            new_pop.append(self._mutate(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(self._mutate(c2))
        self.population = new_pop

    def save_best(self):
        best = max(self.population, key=lambda x: x.fitness)
        record = {
            "generation": self.generation,
            "best_individual": best.to_dict(),
            "avg_fitness": round(sum(i.fitness for i in self.population) / len(self.population), 3),
            "seed": self.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.history.append(record)
        with open("ga_results.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        self._log(f"第{self.generation}代最优已保存 | 得分:{best.fitness:.2f}")

    def run(self):
        self._log(f"GA 搜索启动 | seed={self.seed} | 代数:{MAX_GENERATIONS} | 种群:{POPULATION_SIZE}")
        self.initialize_population()
        self.evaluate_fitness()
        self.save_best()
        for gen in range(1, MAX_GENERATIONS + 1):
            self.generation = gen
            self._log(f"--- 第 {gen}/{MAX_GENERATIONS} 代 ---")
            self.evolve_one_generation()
            self.evaluate_fitness()
            self.save_best()
        final = max(self.population, key=lambda x: x.fitness)
        self._log(f"搜索完成 | 最优: {final.genes} | 得分: {final.fitness:.2f}")
        if "context" in state:
            state["ga_complete"] = True
        return final


def run_ga_phase(query=None, seed=None):
    """运行 GA 发现阶段，返回 MaterialGA 实例。"""
    ga = MaterialGA(query=query, seed=seed)
    ga.run()
    return ga


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or None
    run_ga_phase(query=query)
