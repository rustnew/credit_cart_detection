use csv::Reader;
use  csv::Writer;
use linfa::prelude::*;
use linfa_clustering::Dbscan;
use ndarray::{Array2, Axis};
use plotters::prelude::*;
use serde::Deserialize;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct Transaction {
    time: f64,
    v1: f64,
    v2: f64,
    v3: f64,
    v4: f64,
    v5: f64,
    v6: f64,
    v7: f64,
    v8: f64,
    v9: f64,
    v10: f64,
    v11: f64,
    v12: f64,
    v13: f64,
    v14: f64,
    v15: f64,
    v16: f64,
    v17: f64,
    v18: f64,
    v19: f64,
    v20: f64,
    v21: f64,
    v22: f64,
    v23: f64,
    v24: f64,
    v25: f64,
    v26: f64,
    v27: f64,
    v28: f64,
    amount: f64,
    class: u8,
}

pub fn load_data() -> Result<(Array2<f64>, Vec<u8>), Box<dyn std::error::Error>> {
    let file = File::open("creditcard.csv")?;
    let mut rdr = Reader::from_reader(file);
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.deserialize() {
        let record: Transaction = result?;
        data.push(vec![
            record.time,
            record.v1,
            record.v2,
            record.v3,
            record.v4,
            record.v5,
            record.v6,
            record.v7,
            record.v8,
            record.v9,
            record.v10,
            record.v11,
            record.v12,
            record.v13,
            record.v14,
            record.v15,
            record.v16,
            record.v17,
            record.v18,
            record.v19,
            record.v20,
            record.v21,
            record.v22,
            record.v23,
            record.v24,
            record.v25,
            record.v26,
            record.v27,
            record.v28,
            record.amount,
        ]);
        labels.push(record.class);
    }

    let features = Array2::from(data);
    Ok((features, labels))
}

pub fn normalize_data(data: &Array2<f64>) -> Array2<f64> {
    let mut normalized = data.clone();
    for mut col in normalized.axis_iter_mut(Axis(1)) {
        let mean = col.mean().unwrap_or(0.0);
        let std = (col.var(1.0)).sqrt();
        if std > 0.0 {
            col.mapv_inplace(|x| (x - mean) / std);
        }
    }
    normalized
}

pub fn detect_anomalies_dbscan(data: &Array2<f64>) -> Result<Vec<Option<usize>>, Error> {
    let dataset = DatasetBase::new(data.clone(), Array2::zeros((data.nrows(), 0)));
    let clusters = Dbscan::params(5)
        .tolerance(0.5)
        .fit(&dataset)?;
    Ok(clusters.to_vec())
}

pub fn evaluate_anomalies(clusters: &[Option<usize>], labels: &[u8]) -> (usize, usize, usize) {
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for (cluster, &label) in clusters.iter().zip(labels.iter()) {
        let predicted = if cluster.is_none() { 1 } else { 0 };
        match (predicted, label) {
            (1, 1) => true_positives += 1,
            (1, 0) => false_positives += 1,
            (0, 1) => false_negatives += 1,
            _ => {}
        }
    }

    (true_positives, false_positives, false_negatives)
}

pub fn plot_anomalies(data: &Array2<f64>, labels: &[u8], clusters: &[Option<usize>]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("anomalies.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_time = data.column(0).max().unwrap_or(1.0);
    let max_amount = data.column(data.ncols() - 1).max().unwrap_or(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Détection d'Anomalies (Amount vs Time)", ("sans-serif", 40))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..max_time, 0f64..max_amount)?;

    chart.configure_mesh().draw()?;

    let mut normal_points = Vec::new();
    let mut anomaly_points = Vec::new();

    for (i, row) in data.rows().into_iter().enumerate() {
        let point = (row[0], row[data.ncols() - 1]);
        if clusters[i].is_none() || labels[i] == 1 {
            anomaly_points.push(point);
        } else {
            normal_points.push(point);
        }
    }

    chart.draw_series(normal_points.iter().map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled())))?;
    chart.draw_series(anomaly_points.iter().map(|&(x, y)| Circle::new((x, y), 5, RED.filled())))?;

    root.present()?;
    Ok(())
}

pub fn export_anomalies(data: &Array2<f64>, clusters: &[Option<usize>]) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = Writer::from_path("anomalies.csv")?;
    wtr.write_record(&["time", "amount", "is_anomaly"])?;

    for (i, row) in data.rows().into_iter().enumerate() {
        if clusters[i].is_none() {
            wtr.write_record(&[
                row[0].to_string(),
                row[data.ncols() - 1].to_string(),
                "true".to_string(),
            ])?;
        }
    }

    wtr.flush()?;
    Ok(())
}
