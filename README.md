

# 📊 Détection d'Anomalies de Transactions Bancaires avec DBSCAN

Ce projet utilise l'algorithme DBSCAN pour détecter des anomalies (potentiellement des fraudes) dans un jeu de données de transactions bancaires (`creditcard.csv`). Il s’appuie sur la librairie [linfa](https://github.com/rust-ml/linfa) pour le machine learning et `plotters` pour la visualisation.

## 📁 Données utilisées

Le fichier `creditcard.csv` doit être placé à la racine du projet. Chaque ligne représente une transaction avec :

* `time`: Temps écoulé depuis la première transaction (en secondes)
* `v1` à `v28`: Variables transformées par PCA (caractéristiques anonymisées)
* `amount`: Montant de la transaction
* `class`: 0 pour une transaction normale, 1 pour une fraude

---

## 🧠 Fonctionnalités principales

### 🔍 `load_data`

```rust
pub fn load_data() -> Result<(Array2<f64>, Vec<u8>), Box<dyn std::error::Error>>
```

* Lit le fichier CSV `creditcard.csv`
* Convertit les lignes en `Transaction` via `serde::Deserialize`
* Retourne une matrice de caractéristiques (`Array2<f64>`) et un vecteur d’étiquettes (0 ou 1)

---

### ⚖️ `normalize_data`

```rust
pub fn normalize_data(data: &Array2<f64>) -> Array2<f64>
```

* Normalise chaque colonne (standardisation : moyenne = 0, écart-type = 1)
* Utile pour que DBSCAN fonctionne correctement sans biais dû à l’échelle des valeurs

---

### 🧪 `detect_anomalies_dbscan`

```rust
pub fn detect_anomalies_dbscan(data: &Array2<f64>) -> Result<Vec<Option<usize>>, Error>
```

* Applique l'algorithme DBSCAN sur les données
* Paramètres :

  * `min_points = 5`
  * `tolerance (epsilon) = 0.5`
* Retourne un vecteur d’option :

  * `Some(cluster_id)` pour les points regroupés
  * `None` pour les anomalies détectées (points bruités)

---

### 📈 `evaluate_anomalies`

```rust
pub fn evaluate_anomalies(clusters: &[Option<usize>], labels: &[u8]) -> (usize, usize, usize)
```

* Compare les anomalies détectées par DBSCAN avec les vraies fraudes (`class == 1`)
* Retourne :

  * `true_positives`: anomalies détectées correctes
  * `false_positives`: normales détectées à tort comme anomalies
  * `false_negatives`: fraudes non détectées

---

### 📊 `plot_anomalies`

```rust
pub fn plot_anomalies(data: &Array2<f64>, labels: &[u8], clusters: &[Option<usize>]) -> Result<(), Box<dyn std::error::Error>>
```

* Génère un graphique `anomalies.png`
* Affiche les transactions :

  * Axe X : `time`
  * Axe Y : `amount`
  * Couleurs :

    * 🔵 Bleu : transactions normales
    * 🔴 Rouge : anomalies détectées ou fraudes connues

---

### 📤 `export_anomalies`

```rust
pub fn export_anomalies(data: &Array2<f64>, clusters: &[Option<usize>]) -> Result<(), Box<dyn std::error::Error>>
```

* Exporte les anomalies détectées dans `anomalies.csv`
* Colonnes :

  * `time`, `amount`, `is_anomaly=true`

---

## 🚀 Exemple d'utilisation

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (data, labels) = load_data()?;
    let normalized = normalize_data(&data);
    let clusters = detect_anomalies_dbscan(&normalized)?;

    let (tp, fp, fn_) = evaluate_anomalies(&clusters, &labels);
    println!("TP: {}, FP: {}, FN: {}", tp, fp, fn_);

    plot_anomalies(&data, &labels, &clusters)?;
    export_anomalies(&data, &clusters)?;
    Ok(())
}
```

---

## 📦 Dépendances principales

```toml
[dependencies]
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
ndarray = "0.15"
linfa = "0.6"
linfa-clustering = "0.6"
plotters = "0.3"
```

---

## 📁 Fichiers produits

* `anomalies.png` : Visualisation des anomalies
* `anomalies.csv` : Liste des anomalies détectées (temps + montant)

---

## 🛠️ À améliorer

* Réduction de dimensions (t-SNE, PCA) avant clustering
* Interface Web ou API REST pour tester des fichiers
* Export JSON ou affichage en ligne de commande




