package com.example.tensorflowrecognition.ui.utils

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector
import java.io.File

class ObjectDetectorHelper(
    private val context: Context,
    private val scope: CoroutineScope,
    private val threshold: Float = 0.5f,
    private val numThreads: Int = 2,
    private val maxResults: Int = 3,
) {

    private lateinit var objectDetector: ObjectDetector

    private val _objectDetectorState = MutableSharedFlow<ObjectDetectorState>()
    val objectDetectorState: SharedFlow<ObjectDetectorState>
        get() = _objectDetectorState.asSharedFlow()

    init {
        TfLiteGpu.isGpuDelegateAvailable(context).onSuccessTask { gpuAvailable: Boolean ->
            val optionsBuilder = TfLiteInitializationOptions.builder()
            if (gpuAvailable) {
                optionsBuilder.setEnableGpuDelegateSupport(true)
            }
            TfLiteVision.initialize(context, optionsBuilder.build())
        }.addOnSuccessListener {
            scope.launch {
                _objectDetectorState.emit(ObjectDetectorState.Initialized)
            }
        }.addOnFailureListener {
            scope.launch {
                _objectDetectorState.emit(ObjectDetectorState.Error(it))
            }
        }
    }

    fun setupObjectDetector(modelFile: File) {
        if (!TfLiteVision.isInitialized()) {
            return
        }

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
            .setBaseOptions(baseOptionsBuilder.build())
            .build()

        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(modelFile, options)
        } catch (e: Exception) {
            scope.launch {
                _objectDetectorState.emit(ObjectDetectorState.Error(e))
            }
        }
    }

    fun detect(image: Bitmap, imageRotation: Int, onResults: (ObjectDetectorState.Result) -> Unit) {
        if (!TfLiteVision.isInitialized() || !::objectDetector.isInitialized) {
            return
        }

        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector.detect(tensorImage)
        onResults(ObjectDetectorState.Result(results, tensorImage.height, tensorImage.width))
    }
}
