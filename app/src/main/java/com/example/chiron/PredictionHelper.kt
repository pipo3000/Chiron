package com.example.chiron

import kotlin.math.sqrt

object PredictionHelper {
    /**
     * Makes a prediction based on accelerometer data.
     * This is a placeholder that calculates movement intensity.
     * Replace this with your Cosinorage Python package integration.
     */
    fun predict(accelerometerData: AccelerometerData): String {
        // Calculate magnitude of acceleration (movement intensity)
        val magnitude = sqrt(
            accelerometerData.x * accelerometerData.x +
            accelerometerData.y * accelerometerData.y +
            accelerometerData.z * accelerometerData.z
        )

        // Simple prediction based on magnitude
        // Gravity is approximately 9.8 m/s²
        // When device is still, magnitude ≈ 9.8
        val movementIntensity = magnitude - 9.8f

        return when {
            movementIntensity < 0.5f -> "Still"
            movementIntensity < 2.0f -> "Gentle Movement"
            movementIntensity < 5.0f -> "Moderate Activity"
            movementIntensity < 10.0f -> "Active Movement"
            else -> "Intense Activity"
        }
    }

    /**
     * Get detailed prediction information including the raw values
     */
    fun getPredictionDetails(accelerometerData: AccelerometerData): PredictionDetails {
        val magnitude = sqrt(
            accelerometerData.x * accelerometerData.x +
            accelerometerData.y * accelerometerData.y +
            accelerometerData.z * accelerometerData.z
        )
        val movementIntensity = magnitude - 9.8f

        return PredictionDetails(
            prediction = predict(accelerometerData),
            magnitude = magnitude,
            movementIntensity = movementIntensity,
            x = accelerometerData.x,
            y = accelerometerData.y,
            z = accelerometerData.z
        )
    }
}

data class PredictionDetails(
    val prediction: String,
    val magnitude: Float,
    val movementIntensity: Float,
    val x: Float,
    val y: Float,
    val z: Float
)

