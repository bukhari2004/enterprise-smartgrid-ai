"""Grid Knowledge Graph.

NetworkX is the in-process graph engine (always available). The same graph can be exported to Neo4j
(`to_cypher()` / `export_neo4j()`), which is what the docker-compose deployment uses.

Node types : Region, PowerPlant, TransmissionLine, Substation, Transformer, Meter, Consumer, RenewableAsset,
             MaintenanceTeam, Incident
Relations  : POWERS, FEEDS, SUPPLIES, SERVES, OWNS, INJECTS_INTO, MAINTAINS, AFFECTED, LOCATED_IN
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .simulator import SimData

SUPPLY_RELS = {"POWERS", "FEEDS", "SUPPLIES", "SERVES"}


class GridKnowledgeGraph:
    def __init__(self, data: SimData):
        self.d = data
        self.G = nx.MultiDiGraph()
        self.supply = nx.DiGraph()          # physical power-flow view (plant -> line -> substation -> transformer -> meter)
        self._build()

    # ------------------------------------------------------------ build
    def _node(self, nid, ntype, **attrs):
        self.G.add_node(nid, type=ntype, **attrs)

    def _edge(self, a, b, rel, **attrs):
        self.G.add_edge(a, b, rel=rel, **attrs)
        if rel in SUPPLY_RELS:
            self.supply.add_edge(a, b, rel=rel)

    def _build(self):
        d = self.d
        for r in d.regions.itertuples():
            self._node(r.region_id, "Region", name=r.name)
        for p in d.plants.itertuples():
            self._node(p.plant_id, "PowerPlant", kind=p.type, capacity_mw=p.capacity_mw, region_id=p.region_id)
            self._edge(p.plant_id, p.region_id, "LOCATED_IN")
        for s in d.substations.itertuples():
            self._node(s.substation_id, "Substation", capacity_mva=s.capacity_mva, voltage_kv=int(s.voltage_kv), region_id=s.region_id)
            self._edge(s.substation_id, s.region_id, "LOCATED_IN")
        for l in d.lines.itertuples():
            self._node(l.line_id, "TransmissionLine", capacity_mw=l.capacity_mw, length_km=l.length_km)
            self._edge(l.from_id, l.line_id, "POWERS"); self._edge(l.line_id, l.to_id, "FEEDS")
        for t in d.transformers.itertuples():
            self._node(t.transformer_id, "Transformer", rated_kva=int(t.rated_kva), region_id=t.region_id)
            self._edge(t.substation_id, t.transformer_id, "SUPPLIES")
        for m in d.meters.itertuples():
            self._node(m.meter_id, "Meter", tariff=m.tariff, region_id=m.region_id)
            self._node(m.consumer_id, "Consumer", tariff=m.tariff)
            self._edge(m.transformer_id, m.meter_id, "SERVES"); self._edge(m.consumer_id, m.meter_id, "OWNS")
        subs_by_region = d.substations.groupby("region_id").substation_id.apply(list).to_dict()
        for i, a in enumerate(d.renewables.itertuples()):
            self._node(a.asset_id, "RenewableAsset", kind=a.kind, capacity_mw=a.capacity_mw, region_id=a.region_id)
            cands = subs_by_region.get(a.region_id) or list(d.substations.substation_id)
            self._edge(a.asset_id, cands[i % len(cands)], "INJECTS_INTO")
        for t in d.teams.itertuples():
            self._node(t.team_id, "MaintenanceTeam", crew_size=int(t.crew_size), region_id=t.region_id)
            for s in subs_by_region.get(t.region_id, []):
                self._edge(t.team_id, s, "MAINTAINS")
        for inc in d.incidents.itertuples():
            self._node(inc.incident_id, "Incident", kind=inc.kind, customers_affected=inc.customers_affected, start=str(inc.start))
            if inc.asset_id in self.G:
                self._edge(inc.incident_id, inc.asset_id, "AFFECTED")

    # ------------------------------------------------------------ queries
    def stats(self) -> dict:
        types = pd.Series([a["type"] for _, a in self.G.nodes(data=True)]).value_counts().to_dict()
        rels = pd.Series([a["rel"] for *_, a in self.G.edges(data=True)]).value_counts().to_dict()
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(), "node_types": types, "relationships": rels}

    def node(self, nid: str) -> dict:
        if nid not in self.G:
            raise KeyError(nid)
        out = {"id": nid, **self.G.nodes[nid], "relations": {}}
        for _, v, a in self.G.out_edges(nid, data=True):
            out["relations"].setdefault(f"-[{a['rel']}]->", []).append(v)
        for u, _, a in self.G.in_edges(nid, data=True):
            out["relations"].setdefault(f"<-[{a['rel']}]-", []).append(u)
        for k, v in out["relations"].items():
            if len(v) > 12:
                out["relations"][k] = v[:12] + [f"... (+{len(v) - 12} more)"]
        return out

    def _meters_under(self, nodes) -> set:
        res = set()
        for n in nodes:
            res |= {x for x in nx.descendants(self.supply, n) if self.G.nodes[x]["type"] == "Meter"}
            if self.G.nodes[n]["type"] == "Meter":
                res.add(n)
        return res

    def islanded_substations(self, failed_plants: set[str] | None = None, failed_lines: set[str] | None = None,
                             failed_substations: set[str] | None = None) -> list[str]:
        """Substations with NO remaining path from any healthy plant (N-k contingency analysis)."""
        failed = (failed_plants or set()) | (failed_lines or set()) | (failed_substations or set())
        H = self.supply.copy(); H.remove_nodes_from([n for n in failed if n in H])
        live = set()
        for p in self.d.plants.plant_id:
            if p in H:
                live |= nx.descendants(H, p)
        return [s for s in self.d.substations.substation_id if s not in live]

    def impact(self, asset_id: str) -> dict:
        """Who loses supply if `asset_id` fails? Accounts for N-1 redundancy of dual-fed substations."""
        if asset_id not in self.G:
            raise KeyError(asset_id)
        t = self.G.nodes[asset_id]["type"]
        if t == "PowerPlant":
            lost_subs = self.islanded_substations({asset_id}); roots = lost_subs
        elif t == "TransmissionLine":
            lost_subs = self.islanded_substations(failed_lines={asset_id}); roots = lost_subs
        elif t == "Substation":
            lost_subs = self.islanded_substations(failed_substations={asset_id}); roots = [asset_id] if asset_id in lost_subs else [asset_id]
        elif t in ("Transformer", "Meter"):
            roots, lost_subs = [asset_id], []
        else:
            raise ValueError(f"impact analysis is not defined for {t}")
        meters = self._meters_under(roots)
        mdf = self.d.meters.set_index("meter_id").loc[sorted(meters)] if meters else pd.DataFrame(columns=["tariff", "region_id"])
        redundant = t in ("PowerPlant", "TransmissionLine") and not lost_subs
        return {"asset": asset_id, "type": t, "islanded_substations": lost_subs, "customers_affected": len(meters),
                "industrial_customers": int((mdf.tariff == "industrial").sum()) if len(mdf) else 0,
                "regions": mdf.region_id.value_counts().to_dict() if len(mdf) else {},
                "n_minus_1_secure": bool(redundant),
                "note": "all downstream substations have an alternate feed (N-1 secure)" if redundant else ""}

    def criticality(self, top: int = 10) -> pd.DataFrame:
        """Substation criticality = downstream customers x single-feed penalty."""
        rows = []
        for s in self.d.substations.substation_id:
            n_up = len([u for u in self.supply.predecessors(s)])
            cust = len(self._meters_under([s]))
            rows.append(dict(substation_id=s, feeds=n_up, customers=cust, score=cust * (2.0 if n_up <= 1 else 1.0),
                             single_feed=n_up <= 1))
        return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True).head(top)

    def shortest_path(self, a: str, b: str) -> list[str]:
        return nx.shortest_path(self.supply.to_undirected(as_view=True), a, b)   # along the physical supply chain

    def incidents_for(self, asset_id: str) -> list[dict]:
        return [dict(id=u, **self.G.nodes[u]) for u, _, a in self.G.in_edges(asset_id, data=True) if a["rel"] == "AFFECTED"]

    # ------------------------------------------------------------ GNN-lite risk propagation
    def propagate_risk(self, base_risk: dict[str, float], hops: int = 2, alpha: float = 0.25) -> pd.DataFrame:
        """Graph message passing (GCN-style normalised-adjacency smoothing, no learned weights):
        transformers sharing a substation/feeder influence each other's risk. r <- (1-a) r + a * A_hat r"""
        tr = list(self.d.transformers.transformer_id)
        sub = self.d.transformers.set_index("transformer_id").substation_id
        idx = {t: i for i, t in enumerate(tr)}
        n = len(tr)
        A = np.zeros((n, n))
        for _, grp in sub.groupby(sub):
            ids = [idx[t] for t in grp.index]
            for i in ids:
                for j in ids:
                    if i != j:
                        A[i, j] = 1
        deg = A.sum(1, keepdims=True); A_hat = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
        r0 = np.array([base_risk.get(t, 0.0) for t in tr]); r = r0.copy()
        for _ in range(hops):
            r = (1 - alpha) * r + alpha * (A_hat @ r)
        return pd.DataFrame({"transformer_id": tr, "own_risk": r0, "graph_risk": r, "substation_id": sub.reindex(tr).to_numpy()})

    # ------------------------------------------------------------ Neo4j
    def to_cypher(self, max_nodes: int = 200) -> str:
        """Cypher script that recreates (a sample of) the graph."""
        lines, keep = [], list(self.G.nodes)[:max_nodes]
        ks = set(keep)
        for n in keep:
            a = self.G.nodes[n]; props = ", ".join(f"{k}: {v!r}" for k, v in a.items() if k != "type")
            lines.append(f"MERGE (:{a['type']} {{id: '{n}'{', ' + props if props else ''}}});")
        for u, v, a in self.G.edges(data=True):
            if u in ks and v in ks:
                lines.append(f"MATCH (a {{id:'{u}'}}), (b {{id:'{v}'}}) MERGE (a)-[:{a['rel']}]->(b);")
        return "\n".join(lines)

    def export_neo4j(self, uri: str, user: str, password: str, batch: int = 2000) -> dict:   # pragma: no cover
        from neo4j import GraphDatabase                                    # type: ignore
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as s:
            s.run("CREATE CONSTRAINT grid_id IF NOT EXISTS FOR (n:GridNode) REQUIRE n.id IS UNIQUE")
            nodes = [{"id": n, "type": a["type"], "props": {k: v for k, v in a.items() if k != "type"}} for n, a in self.G.nodes(data=True)]
            for i in range(0, len(nodes), batch):
                s.run("UNWIND $rows AS r CALL apoc.merge.node(['GridNode', r.type], {id: r.id}, r.props) YIELD node RETURN count(node)",
                      rows=nodes[i:i + batch])
            edges = [{"a": u, "b": v, "rel": a["rel"]} for u, v, a in self.G.edges(data=True)]
            for i in range(0, len(edges), batch):
                s.run("UNWIND $rows AS r MATCH (a:GridNode {id:r.a}), (b:GridNode {id:r.b}) "
                      "CALL apoc.merge.relationship(a, r.rel, {}, {}, b) YIELD rel RETURN count(rel)", rows=edges[i:i + batch])
        driver.close()
        return {"nodes": len(nodes), "edges": len(edges)}
