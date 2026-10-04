/**
 * maxflow.js — minimum s-t cut by the Boykov-Kolmogorov algorithm.
 *
 * Y. Boykov and V. Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for
 * Energy Minimization in Vision", PAMI 2004. Two search trees grow from the terminals, meet on an
 * augmenting path, and are repaired (adoption) instead of rebuilt after each augmentation, which is
 * what makes it fast on the sparse, locally connected graphs a mesh gives. The structure follows
 * Kolmogorov's reference maxflow-v3.01: terminal capacities are stored as one signed residual per
 * node, arcs are created in sister pairs (arc a's reverse is a ^ 1).
 *
 * Used by cutFill.js (one fill as a binary cut) and by anything that minimises a pairwise energy
 * over facet labels. Checked against a plain augmenting-path solver on random graphs by
 * test/maxflowCheck.mjs.
 */

const FREE = -1;
const TERMINAL = -2;
const ORPHAN = -3;
const INF_D = 1 << 30;

export class MaxFlow {
  constructor(nodes, arcHint = nodes * 4) {
    this.n = nodes;
    this.trCap = new Float64Array(nodes);
    this.first = new Int32Array(nodes).fill(-1);
    this.head = new Int32Array(arcHint * 2);
    this.nextArc = new Int32Array(arcHint * 2);
    this.rCap = new Float64Array(arcHint * 2);
    this.arcs = 0;
    this.flow = 0;
  }

  /** Terminal links: `source` from the source to i, `sink` from i to the sink. */
  addTweights(i, source, sink) {
    const delta = this.trCap[i];
    if (delta > 0) source += delta; else sink -= delta;
    this.flow += Math.min(source, sink);
    this.trCap[i] = source - sink;
  }

  /** Arc i -> j with capacity `cap`, and j -> i with `rev`. */
  addEdge(i, j, cap, rev) {
    if (this.arcs + 2 > this.head.length) this.#grow();
    const a = this.arcs;
    this.head[a] = j; this.rCap[a] = cap; this.nextArc[a] = this.first[i]; this.first[i] = a;
    this.head[a + 1] = i; this.rCap[a + 1] = rev; this.nextArc[a + 1] = this.first[j]; this.first[j] = a + 1;
    this.arcs += 2;
  }

  #grow() {
    const size = this.head.length * 2;
    const h = new Int32Array(size); h.set(this.head); this.head = h;
    const nx = new Int32Array(size); nx.set(this.nextArc); this.nextArc = nx;
    const r = new Float64Array(size); r.set(this.rCap); this.rCap = r;
  }

  /** Run to completion; returns the maximum flow, which equals the minimum cut. */
  maxflow() {
    const n = this.n;
    const { head, nextArc, rCap, first, trCap } = this;
    const parent = new Int32Array(n).fill(FREE);
    const isSink = new Uint8Array(n);
    const next = new Int32Array(n).fill(-1);
    const ts = new Int32Array(n);
    const dist = new Int32Array(n);
    this.parent = parent;
    this.isSink = isSink;
    // Two FIFO queues of active nodes, as in the reference: next[i] === i marks the last one.
    const qFirst = [-1, -1], qLast = [-1, -1];
    const setActive = (i) => {
      if (next[i] !== -1) return;
      if (qLast[1] !== -1) next[qLast[1]] = i; else qFirst[1] = i;
      qLast[1] = i;
      next[i] = i;
    };
    const nextActive = () => {
      for (;;) {
        let i = qFirst[0];
        if (i === -1) {
          qFirst[0] = i = qFirst[1]; qLast[0] = qLast[1];
          qFirst[1] = qLast[1] = -1;
          if (i === -1) return -1;
        }
        if (next[i] === i) qFirst[0] = qLast[0] = -1; else qFirst[0] = next[i];
        next[i] = -1;
        if (parent[i] !== FREE) return i;
      }
    };
    let time = 0;
    for (let i = 0; i < n; i++) {
      ts[i] = time;
      if (trCap[i] > 0) { isSink[i] = 0; parent[i] = TERMINAL; setActive(i); dist[i] = 1; }
      else if (trCap[i] < 0) { isSink[i] = 1; parent[i] = TERMINAL; setActive(i); dist[i] = 1; }
    }

    const orphans = [];
    let oHead = 0;
    const pushOrphan = (i) => { parent[i] = ORPHAN; orphans.push(i); };

    const augment = (mid) => {
      // mid runs from a source-tree node to a sink-tree node
      let bottleneck = rCap[mid];
      let i = head[mid ^ 1];
      for (;;) {
        const a = parent[i];
        if (a === TERMINAL) break;
        if (bottleneck > rCap[a ^ 1]) bottleneck = rCap[a ^ 1];
        i = head[a];
      }
      if (bottleneck > trCap[i]) bottleneck = trCap[i];
      i = head[mid];
      for (;;) {
        const a = parent[i];
        if (a === TERMINAL) break;
        if (bottleneck > rCap[a]) bottleneck = rCap[a];
        i = head[a];
      }
      if (bottleneck > -trCap[i]) bottleneck = -trCap[i];

      rCap[mid ^ 1] += bottleneck;
      rCap[mid] -= bottleneck;
      i = head[mid ^ 1];
      for (;;) {
        const a = parent[i];
        if (a === TERMINAL) break;
        rCap[a] += bottleneck;
        rCap[a ^ 1] -= bottleneck;
        if (!(rCap[a ^ 1] > 0)) pushOrphan(i);
        i = head[a];
      }
      trCap[i] -= bottleneck;
      if (!(trCap[i] > 0)) pushOrphan(i);
      i = head[mid];
      for (;;) {
        const a = parent[i];
        if (a === TERMINAL) break;
        rCap[a ^ 1] += bottleneck;
        rCap[a] -= bottleneck;
        if (!(rCap[a] > 0)) pushOrphan(i);
        i = head[a];
      }
      trCap[i] += bottleneck;
      if (!(trCap[i] < 0)) pushOrphan(i);
      this.flow += bottleneck;
    };

    const adopt = (i) => {
      const sinkSide = isSink[i];
      let dMin = INF_D, aMin = -1;
      for (let a0 = first[i]; a0 !== -1; a0 = nextArc[a0]) {
        // residual toward i along the tree's direction
        if (!((sinkSide ? rCap[a0] : rCap[a0 ^ 1]) > 0)) continue;
        let j = head[a0];
        if (isSink[j] !== sinkSide || parent[j] === FREE) continue;
        let d = 0;
        for (;;) {
          if (ts[j] === time) { d += dist[j]; break; }
          const a = parent[j];
          d++;
          if (a === TERMINAL) { ts[j] = time; dist[j] = 1; break; }
          if (a === ORPHAN) { d = INF_D; break; }
          j = head[a];
        }
        if (d < INF_D) {
          if (d < dMin) { aMin = a0; dMin = d; }
          for (j = head[a0]; ts[j] !== time; j = head[parent[j]]) { ts[j] = time; dist[j] = d--; }
        }
      }
      if (aMin !== -1) {
        parent[i] = aMin;
        ts[i] = time;
        dist[i] = dMin + 1;
        return;
      }
      parent[i] = FREE;
      for (let a0 = first[i]; a0 !== -1; a0 = nextArc[a0]) {
        const j = head[a0];
        if (isSink[j] !== sinkSide) continue;
        const a = parent[j];
        if (a === FREE) continue;
        if (sinkSide ? rCap[a0] > 0 : rCap[a0 ^ 1] > 0) setActive(j);
        if (a !== TERMINAL && a !== ORPHAN && head[a] === i) pushOrphan(j);
      }
    };

    let current = -1;
    for (;;) {
      let i = current;
      if (i !== -1) {
        next[i] = -1;
        if (parent[i] === FREE) i = -1;
      }
      if (i === -1) {
        i = nextActive();
        if (i === -1) break;
      }
      let mid = -1;
      if (!isSink[i]) {
        for (let a = first[i]; a !== -1; a = nextArc[a]) {
          if (!(rCap[a] > 0)) continue;
          const j = head[a];
          if (parent[j] === FREE) {
            isSink[j] = 0; parent[j] = a ^ 1; ts[j] = ts[i]; dist[j] = dist[i] + 1; setActive(j);
          } else if (isSink[j]) { mid = a; break; }
          else if (ts[j] <= ts[i] && dist[j] > dist[i]) { parent[j] = a ^ 1; ts[j] = ts[i]; dist[j] = dist[i] + 1; }
        }
      } else {
        for (let a = first[i]; a !== -1; a = nextArc[a]) {
          if (!(rCap[a ^ 1] > 0)) continue;
          const j = head[a];
          if (parent[j] === FREE) {
            isSink[j] = 1; parent[j] = a ^ 1; ts[j] = ts[i]; dist[j] = dist[i] + 1; setActive(j);
          } else if (!isSink[j]) { mid = a ^ 1; break; }
          else if (ts[j] <= ts[i] && dist[j] > dist[i]) { parent[j] = a ^ 1; ts[j] = ts[i]; dist[j] = dist[i] + 1; }
        }
      }
      time++;
      if (mid !== -1) {
        next[i] = i;          // keep i active: it may have more paths
        current = i;
        augment(mid);
        while (oHead < orphans.length) {
          const o = orphans[oHead++];
          adopt(o);
        }
        orphans.length = 0;
        oHead = 0;
      } else {
        current = -1;
      }
    }
    return this.flow;
  }

  /** After maxflow(): true when node i is on the SOURCE side of the minimum cut. */
  onSource(i) {
    return this.parent[i] !== FREE && !this.isSink[i];
  }

  /** After maxflow(): true when node i is in the SINK tree. A free node (in neither tree) can go
   *  either way at equal cost, and is on neither side by this test or onSource. */
  onSinkSide(i) {
    return this.parent[i] !== FREE && this.isSink[i] === 1;
  }
}
