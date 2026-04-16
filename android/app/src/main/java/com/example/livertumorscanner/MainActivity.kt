package com.example.livertumorscanner

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.view.LayoutInflater
import android.view.View
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.bumptech.glide.Glide
import com.example.livertumorscanner.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val viewModel: AnomalyViewModel by viewModels()

    private val models = arrayOf("conv_ae", "ae_flow", "masked_ae", "ccb_aae", "qformer", "ensemble", "attention_ae", "ALL MODELS")

    private val selectDicomLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            val selectedModel = binding.spinnerModels.selectedItem.toString()
            viewModel.uploadDicomFile(uri, selectedModel)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val adapter = ArrayAdapter(this, R.layout.spinner_item, models)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModels.adapter = adapter
        binding.spinnerModels.setSelection(2) // default to masked_ae

        binding.btnUpload.setOnClickListener {
            // Allows uploading standard DICOM or ZIP folders Native!
            selectDicomLauncher.launch("*/*")
        }

        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.isLoading.observe(this) { loading ->
            binding.loadingLayout.visibility = if (loading) View.VISIBLE else View.GONE
            binding.btnUpload.isEnabled = !loading
        }

        viewModel.errorState.observe(this) { error ->
            Toast.makeText(this, error, Toast.LENGTH_LONG).show()
        }

        // Single Model View
        viewModel.singlePredictionResult.observe(this) { result ->
            if (result == null) return@observe
            binding.llResultsContainer.removeAllViews()
            binding.tvResultLabel.visibility = View.VISIBLE
            binding.tvResultLabel.text = "DIAGNOSTIC REPORT: ${result.label.uppercase()}"
            binding.tvResultLabel.setTextColor(getColorForLabel(result.label))
            
            addModelCard(result.model, result)
        }

        // Compare View
        viewModel.comparePredictionResult.observe(this) { result ->
            if (result == null) return@observe
            binding.llResultsContainer.removeAllViews()
            binding.tvResultLabel.visibility = View.VISIBLE
            binding.tvResultLabel.text = "CONSENSUS REPORT: ${result.majority_vote.uppercase()}"
            binding.tvResultLabel.setTextColor(getColorForLabel(result.majority_vote))

            result.model_results.forEach { (modelName, prediction) ->
                if (prediction.images != null) {
                    addModelCard(modelName, prediction)
                }
            }
        }
    }

    private fun addModelCard(modelName: String, result: PredictionResponse) {
        val cardView = LayoutInflater.from(this).inflate(R.layout.item_model_result, binding.llResultsContainer, false)
        
        val tvName = cardView.findViewById<TextView>(R.id.tvModelName)
        val tvScore = cardView.findViewById<TextView>(R.id.tvModelScore)
        val tvLabel = cardView.findViewById<TextView>(R.id.tvModelLabel)
        val badgeStatus = cardView.findViewById<View>(R.id.badgeStatus)
        val ivViewer = cardView.findViewById<ImageView>(R.id.ivViewer)
        val ivViewerSecondary = cardView.findViewById<ImageView>(R.id.ivViewerSecondary)
        val frameSecondary = cardView.findViewById<View>(R.id.frameSecondary)
        val ivDivider = cardView.findViewById<View>(R.id.ivDivider)
        val tvFrameLabel = cardView.findViewById<TextView>(R.id.tvFrameLabel)
        val tvFrameLabelSec = cardView.findViewById<TextView>(R.id.tvFrameLabelSecondary)
        val toggleGroup = cardView.findViewById<com.google.android.material.button.MaterialButtonToggleGroup>(R.id.toggleGroup)

        tvName.text = modelName.replace("_", " ").uppercase()
        tvScore.text = "CONFIDENCE SCORE: ${String.format("%.3f", result.score)}"
        tvLabel.text = result.label.uppercase()
        
        val diagColor = getColorForLabel(result.label)
        tvLabel.setTextColor(diagColor)
        badgeStatus.backgroundTintList = android.content.res.ColorStateList.valueOf(diagColor).withAlpha(40)

        // Initial Logic and Setup
        ivDivider.visibility = View.VISIBLE
        frameSecondary.visibility = View.VISIBLE
        decodeAndBind(result.images.preprocessed, ivViewer)
        decodeAndBind(result.images.overlay, ivViewerSecondary)
        tvFrameLabel.text = "1. ANATOMICAL BASIS (SCAN)"
        tvFrameLabelSec.text = "2. AI TUMOR MASK (DETECTION)"
        toggleGroup.check(R.id.btnOverlay)

        toggleGroup.addOnButtonCheckedListener { group, checkedId, isChecked ->
            if (isChecked) {
                // Force clear previous images to prevent 'ghosting' or 'sticky' views
                Glide.with(this).clear(ivViewer)
                Glide.with(this).clear(ivViewerSecondary)
                
                when (checkedId) {
                    R.id.btnOriginal -> {
                        ivDivider.visibility = View.GONE
                        frameSecondary.visibility = View.GONE
                        decodeAndBind(result.images.original, ivViewer)
                        tvFrameLabel.text = "ORIGINAL DICOM BASIS"
                    }
                    R.id.btnOverlay -> {
                        ivDivider.visibility = View.VISIBLE
                        frameSecondary.visibility = View.VISIBLE
                        decodeAndBind(result.images.preprocessed, ivViewer)
                        decodeAndBind(result.images.overlay, ivViewerSecondary)
                        tvFrameLabel.text = "1. ANATOMICAL BASIS (SCAN)"
                        tvFrameLabelSec.text = "2. AI TUMOR MASK (DETECTION)"
                    }
                    R.id.btnRecon -> {
                        ivDivider.visibility = View.GONE
                        frameSecondary.visibility = View.GONE
                        decodeAndBind(result.images.reconstruction, ivViewer)
                        tvFrameLabel.text = "AI HEALTHY RECONSTRUCTION"
                    }
                }
            }
        }

        binding.llResultsContainer.addView(cardView)
    }

    private fun getColorForLabel(label: String): Int {
        return if (label == "tumor") getColor(R.color.diagnostic_red)
        else getColor(R.color.diagnostic_green)
    }

    private fun decodeAndBind(base64Str: String, targetView: ImageView) {
        try {
            val bytes = Base64.decode(base64Str, Base64.DEFAULT)
            if (bytes.isNotEmpty()) {
                Glide.with(this)
                    .asGif() // Ensure it treats the stream as a gif
                    .load(bytes)
                    .diskCacheStrategy(com.bumptech.glide.load.engine.DiskCacheStrategy.NONE)
                    .skipMemoryCache(true)
                    .into(targetView)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}