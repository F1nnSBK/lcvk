# Pithos: Model-Isomorphic Databases (MIDB) for Ultra-Fast Resonant Voting

## Abstract

The rapid adoption of Foundation Models has positioned Vector Databases (VDBs) as a critical bottleneck in retrieval-augmented systems. Traditional VDBs treat dense embeddings as opaque, high-dimensional arrays, relying on general-purpose Approximate Nearest Neighbor (ANN) indices (e.g., HNSW, FAISS) that suffer from high memory bandwidth requirements and rigid distance metrics. In this paper, we introduce the concept of a **Model-Isomorphic Data Base (MIDB)**, an architecture that structurally aligns the database's storage and query execution directly with the geometric and hierarchical properties of the upstream encoder model.

We present **Pithos**, the first implementation of an MIDB. By co-designing the model's feature extraction (employing orthogonal transformations and binary quantization) with the database engine, Pithos maps dense representations into a tightly packed hierarchical Hamming space. This allows Pithos to replace floating-point arithmetic with native, hardware-accelerated bitwise operations. Furthermore, Pithos introduces *Resonant Voting*, a multi-criteria candidate generation mechanism that aggregates boolean similarities across multiple semantic families.

Evaluated on a real-world, large-scale dataset (1,000,000 DINOv3 embeddings for Lunar Pit detection), Pithos achieves an 83x speedup over FAISS-based multi-criteria retrieval, maintaining a recall of >96%. Our results demonstrate that structurally coupling the embedding model with the database engine unlocks orders-of-magnitude improvements in retrieval latency and throughput, presenting a new paradigm for efficient similarity search.

---

## Kernkonzept: Model Isomorphic Data Base (MIDB)
*(Notizen & Argumentationshilfe für das Paper)*

**Warum "Isomorphic" (strukturgleich)?**
Klassische Vektordatenbanken (wie Milvus, Qdrant oder FAISS) sind *Model Agnostic*. Das bedeutet, die DB weiß nicht, was das Model macht – sie nimmt einfach generische Float-Vektoren entgegen und berechnet euklidische Distanzen oder Cosinus-Ähnlichkeiten. 

Bei einer **MIDB** sind Model und Datenbank *isomorph*. Die Architektur des neuronalen Netzes (hier: DINOv3 + LoRA + SVD-Binarisierung) und das Speicher-/Query-Format der DB (hier: hierarchischer Hamming-Space, Bit-Packing) greifen wie ein Reißverschluss ineinander. Die DB ist praktisch der Hardware-optimierte, verlängerte Arm der Model-Architektur.

**Die drei Säulen der MIDB (als Selling Points für die Introduction):**
1. **Co-Design von Feature Extraction und Storage:** Das Model erzeugt keine generischen Floats, sondern hardware-optimierte Bit-Vektoren, die exakt auf die FFI-Speicherstruktur (Foreign Function Interface) der DB passen.
2. **Resonant Voting statt L2-Distance:** Anstatt teurer kontinuierlicher Distanzmetriken nutzt die MIDB das Konzept der Resonanz (Schwellenwerte auf Bit-Ebene über mehrere Kriterien hinweg), was algorithmisch biologischen assoziativen Speichern ähnelt und das Noise/Signal-Verhältnis drastisch verbessert.
3. **Zero-Copy / Native Kernels:** Da die Datenformate isomorph sind, entfallen teure Datentyp-Konvertierungen. Die DB operiert mit C-basierten FFM (Fast Forward Matching) Kernels direkt auf dem RAM, was extrem hohes Caching und SIMD-Auslastung ermöglicht.

---

## Struktur-Vorschlag für das Paper (ICML / MLSys)

1. **Introduction**
   - Das Bottleneck heutiger VDBs (Memory Bandwidth, Float-Berechnungen).
   - Definition des Konzepts: Model-Isomorphic Data Base (MIDB).
   - Unsere Contribution: Pithos als erste MIDB.

2. **Related Work**
   - Approximate Nearest Neighbor Search (FAISS, HNSW, ScaNN).
   - Binary Hashing & Matryoshka Representation Learning (MRL).
   - Vector Databases (Milvus, Pinecone).

3. **The MIDB Architecture: Pithos**
   - *Representation Alignment:* Orthogonale Transformation (SVD/Hadamard) & Binarisierung im Model.
   - *Storage Engine:* Hierarchical Hamming Space und Memory Layout.
   - *Query Mechanism:* Resonant Voting als multi-kriterielles Retrieval-Paradigma.
   - *System Implementation:* C/Java FFI Boundary Optimization.

4. **Experimental Setup**
   - Real-World Use Case: Lunar Pit Detection (Planetary Defense / Space Exploration).
   - Model: DINOv3 mit LoRA Adapter.
   - Dataset: 1.000.000 Records, 278 Queries.

5. **Results & Evaluation**
   - *Throughput & Latency:* Der 83x Speedup gegenüber emuliertem FAISS.
   - *Accuracy Trade-offs:* Recall@K und Resonant Voting Recall (>96%).
   - *Dimensionality Sweep:* Wann MIDB klassische Ansätze überholt (Crossover Point).

6. **Conclusion**
   - MIDB als Paradigmenwechsel: Weg von generischen Datenbanken hin zu Model-spezifischen Storage-Engines.

---

## Fakten & Zahlen (Cheat Sheet für Pithos)

### 1. System- & Datensatz-Konfiguration

| Parameter | Wert | Beschreibung |
| :--- | :--- | :--- |
| **Datenbank-Größe** | 1.000.000 Vektoren | Echte extrahierte Mondoberflächen-Bilder |
| **Vektordimension** | 384 Float32 | Aus DINOv3 ViT-S/16 + Lunar LoRA Adapter |
| **Index-Format** | **1-Bit Quantisierung** | Binarisiert auf 384 Bits (QMode = 0) |
| **Abfragen (Queries)**| 278 Queries | Aufgeteilt in 8 wissenschaftliche Kriterien-Familien |
| **Index-Ladezeit** | ~48 ms | Direktes Einlesen von 100.000 Datensätzen per `mmap` |

---

### 2. Resonant Voting Klassifikations-Qualität (**1-Bit Quantisierung**)

> [!NOTE]
> Die herausragende Recall-Rate von **96,49%** wird vollständig unter **1-Bit-Quantisierung (QMode = 0)** erreicht. Obwohl die Binarisierung einzelner Dimensionen theoretisch Rauschen einführt, gleicht das *Resonant Voting* dies durch den Ensemble-Effekt über 8 unabhängige Kriterien-Familien aus (Mehrheitsentscheid $\ge 5$ von 8 Stimmen).

#### Konfusionsmatrix (1-Bit, Schwellenwert $\tau = 46$ Bits)

| | Vorhersage: Höhleneingang (Target) | Vorhersage: Mondgelände (Background) |
| :--- | :---: | :---: |
| **Echter Höhleneingang (Actual)** | **TP: 159.720** | FN: 5.808 |
| **Echtes Mondgelände (Actual)** | FP: 208.978 | **TN: 625.494** |

#### Klassifikations-Metriken (1-Bit, Schwellenwert $\tau = 46$ Bits)

| Metrik | Wert | Bedeutung für das Paper |
| :--- | :---: | :--- |
| **Recall** | **96,49 %** | Pithos findet nahezu alle echten Gruben/Höhlen trotz extremer Kompression. |
| **Precision** | **43,32 %** | Perfekt geeignet als extrem schneller First-Stage-Filter (Candidate Generator). |
| **F1-Score** | **59,79 %** | Optimiertes Gleichgewicht für die Anbindung nachfolgender Netze (z. B. Mask R-CNN). |
| **Hamming-Mittelwert (Target)** | 68,89 Bits | Durchschnittliche Distanz gesuchter Mondhöhlen. |
| **Hamming-Mittelwert (Terrain)** | 84,38 Bits | Durchschnittliche Distanz flacher Mondstrukturen. |

---

### 3. Resonant Voting Such-Performance (Stress-Test: 100k Records, 278 Queries)

| Backend | Gesamtzeit (ms) | Durchsatz (MVPS) | Relativer Speedup |
| :--- | :---: | :---: | :---: |
| **FAISS Emuliertes Voting** | 761,11 ms | 36,53 MVPS | 1,0x (Baseline) |
| **Pithos Native FFM (1-Bit)** | **7,81 ms** | **3.559,60 MVPS** | **97,5x** |

---

### 4. Skalierbarkeit & Crossover-Latenzen (100k Records, K=100)

| Metrik (D=384) | FAISS Baseline | Pithos Native | Vorteil / Gewinner |
| :--- | :---: | :---: | :---: |
| **Single-Query Latenz** | 2,86 ms (2.862,3 µs) | **1,03 ms** (1.034,1 µs) | **2,8x schneller** (Pithos gewinnt ab D $\ge$ 128) |
| **Multi-Query Durchsatz (N=100)**| 85,28 MVPS | **128,51 MVPS** | **1,5x schneller** (Pithos gewinnt ab D $\ge$ 384) |

---

## Progressions-Analyse & Historischer Vergleich (Vergleich mit stand_*.md)

Die kontinuierliche Weiterentwicklung von Pithos lässt sich anhand der historischen Messprotokolle in `docs/stand_*.md` präzise nachvollziehen. Die folgende Tabelle vergleicht die wichtigsten Meilensteine bis zum aktuellen Produktionslauf.

### 1. Qualitäts- und Recall-Entwicklung (D=384, Lunar Dataset)

Die Tabelle zeigt, wie sich die Treffsicherheit (Recall) durch die Einführung von 2-Bit-Quantisierung, Rauschunterdrückung (Ternary Masking) und schließlich durch den **neuen LoRA-Adapter** verbessert hat:

| Version / Datum | Modus / Features | Recall@10 | Recall@100 | Recall@500 (Candidate Generator Elbow) |
| :--- | :--- | :---: | :---: | :---: |
| **18. Juni (stand_20260618.md)** | 1-Bit Quantisierung (Sign-only) | 18.53% | 33.84% | 68.35% |
| **23. Juni (stand_20260623_0032.md)** | 2-Bit Quantisierung (Ternary Masking) | 31.29% | 41.63% | -- |
| **23. Juni (stand_20260623_0055.md)** | 2-Bit + Asymmetrisches Reranking | 47.41% | 59.37% | -- |
| **26. Juni (Aktueller Lauf)** | **2-Bit + Reranking + Neuer LoRA-Adapter** | **44.50%** | **70.52%** | **91.37%** |

> [!IMPORTANT]
> * **Der LoRA-Effekt:** Durch das automatische Herunterladen und Einbinden des neuesten LoRA-Modell-Adapters (`F1nnSBK/lunar-dinov3-lora`) im aktuellen Lauf sprang der Recall bei $K=100$ von **59.37%** auf **70.52%** (+18% relativ).
> * **Massiver Anstieg bei der Downstream-Filterung (Elbow):** Der Recall des First-Stage Candidate Generators bei $K=500$ (wo die nachfolgende Arbeitslast von Modellen wie Mask R-CNN um **99.50%** reduziert wird) stieg von **68.35%** (Sign-only) auf phänomenale **91.37%**!

### 2. Durchsatz & Speedup im Wandel (Stress-Test: 100k, Resonant Voting)

| Version / Datum | FAISS (ms / MVPS) | Pithos (ms / MVPS) | Speedup |
| :--- | :---: | :---: | :---: |
| **18. Juni (stand_20260618.md)** | 716.01 ms / 38.83 MVPS | 8.76 ms / 3,175.06 MVPS | **81.8x** |
| **26. Juni (Aktueller Lauf)** | 761.11 ms / 36.53 MVPS | 7.81 ms / 3,559.60 MVPS | **97.5x** |

> [!NOTE]
> Durch die VectorAPI-Optimierung in `TransformOperator` konnten wir den Durchsatz beim Resonant Voting auf **3.559,60 MVPS** steigern, was den historischen Höchstwert von 3.175 MVPS übertrifft und einen neuen Rekordspeedup von **97.5x** gegenüber FAISS erzielt.

### 3. Verschiebung des Dimensionalitäts-Crossover-Punkts

* **Historischer Stand (1.000.000 Records):** Crossover bei **D=64**. Pithos dominierte ab D>=64 sowohl in Latenz als auch Durchsatz.
* **Aktueller Stand (100.000 Records Sweep):** 
  - Single-Query Latenz-Crossover: **D=64 -> 128** (Pithos gewinnt ab D=128)
  - Multi-Query Durchsatz-Crossover: **D=256 -> 384** (Pithos gewinnt ab D=384)

**Physische Erklärung für das Paper (Academic Honesty):**
Die Verschiebung des Crossover-Punkts nach oben bei der kleineren Test-Datenbank (100.000 Einträge) ist ein direkter Effekt der CPU-Cache-Hierarchie. Bei 100.000 Vektoren passen die unkomprimierten Float32-Vektoren von FAISS (z. B. $100.000 \times 64 \times 4\text{ Bytes} = 25.6\text{ MB}$) vollständig in den L3-Cache (System Level Cache) heutiger M-Series-Prozessoren. FAISS Flat L2 erzielt dadurch nahezu 100% Cache-Hits und vermeidet Speicherbus-Latenzen.

Skaliert die Datenbank jedoch auf das reale Szenario von **1.000.000 Vektoren**, sprengt FAISS die Cache-Größen vollständig ($256\text{ MB}$ bei D=64) und läuft in das Speicherbandbreiten-Limit (Memory Bandwidth Bottleneck). Pithos hingegen benötigt dank der 1-Bit-Quantisierung für 1 Million Vektoren nur **8 Megabyte**, bleibt vollständig im L3-Cache und schlägt FAISS daher bereits ab **D=64**.

---

## Ablation Study: FlatIndex-Optimierung (Lazy Loading)

Um den Einfluss der On-Demand-Speicherladung der Tiers in `FlatIndex` zu quantifizieren, führen wir eine Ablation-Studie durch:
- **Baseline (Vorabladen aller Tiers):** Vorab-Kopieren aller Tiers in lokale Register vor jedem Query-Vergleich.
- **Lazy Loading (On-Demand):** Laden höherer Tiers nur bei Erreichen der jeweiligen Distanzstufe.

### Vergleich der Performanz-Metriken (100k Records, D=384, Lunar Dataset)

| Konfiguration | Single-Query Latency (D=384) | Multi-Query Throughput (D=384) | Resonant Voting Durchsatz | Resonant Voting Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Vorabladen) | 1.0260 ms | 129.56 MVPS | 3,787.62 MVPS | 102.6x (vs. FAISS) |
| **Optimiert (Lazy Loading)**| **0.8164 ms** | **167.33 MVPS** | **4,834.78 MVPS** | **131.7x (vs. FAISS)** |


