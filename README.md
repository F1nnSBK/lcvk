# Java 25 & GraalVM AOT-Compiled Vector Database

Dieses Repository enthält das Grundgerüst für eine eigene, hochperformante **Vektordatenbank**, geschrieben in **Java 25**, optimiert mit **SIMD (Vector API)** und AOT-kompiliert zu einer nativen Shared Library (`.so` / ELF-Format) mittels **GraalVM Native Image**.

Dank der vollständigen Dockerisierung kann die Linux-Bibliothek (`.so`) auch direkt unter macOS gebaut und in einer Container-Umgebung verifiziert werden.

---

## 🚀 Features

- **Java 25 Ready:** Nutzt moderne Java-Features und Sprachstandards.
- **Hardware-SIMD Beschleunigung:** Verwendet die JVM **Vector API** (`jdk.incubator.vector`) zur Vektorisierung von Distanzberechnungen (Euclidean Distance, Cosine Similarity und Dot Product) direkt auf CPU-Registern (AVX2, AVX-512, Neon).
- **AOT-kompiliert:** Kompiliert zu nativem Maschinencode mittels GraalVM Native Image (keine JVM-Startzeit, minimaler RAM-Bedarf).
- **Native C-Schnittstelle:** Exportiert C-Entrypoints (`@CEntryPoint`), um die Datenbank direkt aus C, C++, Rust, Python oder Go über FFI aufzurufen.
- **Docker-native Pipeline:** Einfaches Bauen der `.so`-Bibliothek und automatisches Testen über Docker.

---

## 📁 Ordnerstruktur

```
.
├── .gitignore          # Git-Ausschlussregeln für Java, Maven und GraalVM
├── Dockerfile          # Multi-stage Build mit GraalVM JDK 25 und GCC
├── README.md           # Projektdokumentation (diese Datei)
├── build.sh            # Skript zum Bauen & Exportieren der .so-Datei
├── pom.xml             # Maven Projektkonfiguration (Compiler- & Native-Plugins)
├── test_client.c       # C-Testclient zum Validieren der Shared Library
├── verify.sh           # Testet den C-Client und die .so-Datei im Container
└── src
    ├── main
    │   └── java
    │       └── org
    │           └── example
    │               └── vectordb
    │                   ├── CApi.java           # GraalVM C-Entrypoints (Brücke zu C)
    │                   ├── DistanceMetric.java # SIMD-beschleunigte Distanzberechnungen
    │                   ├── FlatIndex.java      # Thread-sicherer Linear-Scan Index (Brute-Force)
    │                   ├── Index.java          # Schnittstellen-Definition für Indizes
    │                   ├── VectorDb.java       # Hauptkoordinator der Datenbank
    │                   └── VectorRecord.java   # Immutable Datensatz-Klasse (Java Record)
    └── test
        └── java
            └── org
                └── example
                    └── vectordb
                        └── VectorDbTest.java   # JUnit 5 Tests für Distanzen und Indizes
```

---

## 🛠️ Voraussetzungen

Das einzige Werkzeug, das zwingend auf deinem Host-System installiert sein muss, ist **Docker**. Alle weiteren Abhängigkeiten (JDK 25, GraalVM Compiler, GCC, Maven) werden automatisch innerhalb des Docker-Containers verwaltet.

---

## 📦 Build & Ausführung

### 1. Kompilieren & Shared Library exportieren
Führe das Build-Skript aus. Es baut die Java-Anwendung, führt die Java-Unittests aus, kompiliert das native Image und speichert die fertigen Dateien im Ordner `build-output/` ab:

```bash
./build.sh
```

Nach erfolgreichem Build findest du folgende Dateien in `build-output/`:
- `libvectordb.so` (Die native Linux ELF-Shared-Library)
- `vdb_lib.h` (Die C-Headerdatei mit den Methodendeklarationen)

### 2. Funktion des C-Clients verifizieren
Um die Funktionsweise der `.so`-Datei und die Anbindung an C zu testen (auch wenn du auf macOS arbeitest), kannst du das Testskript ausführen:

```bash
./verify.sh
```

Dieses Skript baut ein schlankes Linux-Image, kompiliert `test_client.c` gegen `libvectordb.so` und führt das Programm aus.

---

## 🔌 C-API Dokumentation

Die Shared Library stellt folgende Kernfunktionen über die Schnittstelle bereit:

```c
// Erstellt ein GraalVM Isolat (Ausführungskontext für Java-Code)
int graal_create_isolate(graal_isolate_params_t* params, graal_isolate_t** isolate, graal_isolatethead_t** thread);

// Initialisiert die Vektordatenbank (muss als erstes aufgerufen werden)
int vdb_init(graal_isolatethead_t* thread);

// Erstellt einen Index
// metricType: 0 = EUCLIDEAN, 1 = DOT_PRODUCT, 2 = COSINE
int vdb_create_index(graal_isolatethead_t* thread, char* name, int dimension, int metricType);

// Fügt einen Vektor hinzu
int vdb_insert(graal_isolatethead_t* thread, char* indexName, long long id, float* values, int length);

// Führt eine K-Nearest-Neighbor-Suche aus
// outIds und outScores müssen Arrays der Größe K sein.
// Gibt bei Erfolg die Anzahl gefundener Vektoren zurück (<= K).
int vdb_search(graal_isolatethead_t* thread, char* indexName, float* query, int length, int k, long long* outIds, float* outScores);

// Gibt die Anzahl der Vektoren im Index zurück
int vdb_size(graal_isolatethead_t* thread, char* indexName);

// Zerstört das Isolat und gibt alle Ressourcen frei
int graal_tear_down_isolate(graal_isolatethead_t* thread);
```

---

## ⚡ SIMD details (Vector API)
In `DistanceMetric.java` wird die hardwarenahe Beschleunigung wie folgt deklariert:

```java
var acc = FloatVector.zero(SPECIES);
int limit = SPECIES.loopBound(a.length);
int i = 0;
for (; i < limit; i += SPECIES.length()) {
    var va = FloatVector.fromArray(SPECIES, a, i);
    var vb = FloatVector.fromArray(SPECIES, b, i);
    acc = va.fma(vb, acc); // va * vb + acc in einem Takt
}
```

Durch das Flag `--add-modules=jdk.incubator.vector` wird diese API sowohl beim Testen als auch im nativen Kompilationsschritt freigeschaltet.
