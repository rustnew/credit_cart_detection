mod  data;
use  data::*;



fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Charger et normaliser les données
    let (data, labels) = load_data()?;
    let normalized_data = normalize_data(&data);

    // Détecter les anomalies avec DBSCAN
    let clusters = detect_anomalies_dbscan(&normalized_data)?;
    println!("Nombre de points non clusterisés (anomalies) : {}", clusters.iter().filter(|c| c.is_none()).count());

    // Évaluer les résultats
    let (tp, fp, fn_) = evaluate_anomalies(&clusters, &labels);
    println!("Vrais positifs: {}, Faux positifs: {}, Faux négatifs: {}", tp, fp, fn_);

    // Visualiser les résultats
    plot_anomalies(&data, &labels, &clusters)?;

    // Exporter les anomalies
    export_anomalies(&data, &clusters)?;

    Ok(())
}