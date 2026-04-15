package com.example.livertumorscanner

data class PredictionResponse(
    val score: Double,
    val label: String,
    val model: String = "",
    val threshold: Double = 0.0,
    val images: ImagePayload
)

data class ImagePayload(
    val original: String,
    val preprocessed: String,
    val reconstruction: String,
    val error_map: String,
    val overlay: String
)

data class CompareResponse(
    val majority_vote: String,
    val model_results: Map<String, PredictionResponse>
)
