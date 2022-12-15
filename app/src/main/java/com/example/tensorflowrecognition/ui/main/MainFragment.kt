package com.example.tensorflowrecognition.ui.main

import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.tensorflowrecognition.R
import com.example.tensorflowrecognition.databinding.FragmentMainBinding
import com.example.tensorflowrecognition.ui.permission.PermissionsFragment
import com.example.tensorflowrecognition.ui.utils.ObjectDetectorHelper
import com.example.tensorflowrecognition.ui.utils.ObjectDetectorState
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onEach
import org.tensorflow.lite.task.gms.vision.detector.Detection
import java.io.File
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainFragment : Fragment(R.layout.fragment_main) {

    companion object {
        private const val FIREBASE_MODEL_NAME = "object-detection"
    }

    private val binding by viewBinding(FragmentMainBinding::bind)

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var cameraExecutor: ExecutorService
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        observeObjectDetector()
    }

    override fun onResume() {
        super.onResume()
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(MainFragmentDirections.actionMainToPermissions())
        }
    }

    private fun observeObjectDetector() {
        objectDetectorHelper = ObjectDetectorHelper(requireContext(), viewLifecycleOwner.lifecycleScope)
        viewLifecycleOwner.lifecycleScope.launchWhenStarted {
            objectDetectorHelper.objectDetectorState
                .onEach { objectDetectorState ->
                    when (objectDetectorState) {
                        ObjectDetectorState.Initialized -> downloadFirebaseModel()
                        is ObjectDetectorState.Error -> objectDetectorState.throwable.localizedMessage?.let(::onError)
                        is ObjectDetectorState.Result -> onResults(
                            objectDetectorState.results,
                            objectDetectorState.imageWidth,
                            objectDetectorState.imageHeight
                        )
                    }
                }
                .collect()
        }
    }

    private fun downloadFirebaseModel() {
        val conditions = CustomModelDownloadConditions.Builder().build()
        FirebaseModelDownloader.getInstance()
            .getModel(FIREBASE_MODEL_NAME, DownloadType.LOCAL_MODEL, conditions)
            .addOnSuccessListener {
                it.file?.let(::onInitialized)
            }.addOnFailureListener {
                it.localizedMessage?.let(::onError)
            }
    }

    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()
                bindCamera()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    private fun bindCamera() {

        // CameraProvider
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(binding.viewFinder.display.rotation)
                .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(binding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                        }

                        detectObjects(image)
                    }
                }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (e: Exception) {
            e.localizedMessage?.let(::onError)
        }
    }

    private fun detectObjects(image: ImageProxy) {
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation) {
            onResults(it.results, bitmapBuffer.width, bitmapBuffer.height)
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = binding.viewFinder.display.rotation
    }

    private fun onInitialized(customModelFile: File) {
        objectDetectorHelper.setupObjectDetector(customModelFile)
        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.viewFinder.post {
            setUpCamera()
        }
        binding.progressCircular.isVisible = false
    }

    private fun onError(error: String) {
        Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        binding.progressCircular.isVisible = false
    }

    private fun onResults(
        results: MutableList<Detection>?,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            binding.overlay.setResults(results ?: LinkedList<Detection>(), imageHeight, imageWidth)
        }
    }
}