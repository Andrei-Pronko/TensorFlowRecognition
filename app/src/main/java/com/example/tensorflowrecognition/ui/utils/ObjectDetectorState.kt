package com.example.tensorflowrecognition.ui.utils

import org.tensorflow.lite.task.gms.vision.detector.Detection

sealed interface ObjectDetectorState {
    object Initialized : ObjectDetectorState
    data class Error(val throwable: Throwable) : ObjectDetectorState
    data class Result(
        val results: MutableList<Detection>?,
        val imageHeight: Int,
        val imageWidth: Int,
    ) : ObjectDetectorState
}