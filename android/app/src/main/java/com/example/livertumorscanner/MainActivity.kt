package com.example.livertumorscanner

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
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

    private val models = arrayOf("conv_ae", "ae_flow", "masked_ae", "ccb_aae", "qformer", "ensemble", "ALL MODELS")

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

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, models)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModels.adapter = adapter
        binding.spinnerModels.setSelection(5) // default to ensemble

        binding.btnUpload.setOnClickListener {
            // Allows uploading standard DICOM or ZIP folders Native!
            selectDicomLauncher.launch("*/*")
        }

        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.isLoading.observe(this) { loading ->
            binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
            binding.btnUpload.isEnabled = !loading
        }

        viewModel.errorState.observe(this) { error ->
            Toast.makeText(this, error, Toast.LENGTH_LONG).show()
        }

        // Single Model View
        viewModel.singlePredictionResult.observe(this) { result ->
            if (result == null) return@observe
            binding.llResultsContainer.removeAllViews()
            binding.tvResultLabel.text = "Volume Analysis: ${result.label.uppercase()}"
            binding.tvResultLabel.setTextColor(getColorForLabel(result.label))
            
            addModelCard(result.model, result)
        }

        // Compare View
        viewModel.comparePredictionResult.observe(this) { result ->
            if (result == null) return@observe
            binding.llResultsContainer.removeAllViews()
            binding.tvResultLabel.text = "Majority Consensus: ${result.majority_vote.uppercase()}"
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
        val ivOrig = cardView.findViewById<ImageView>(R.id.ivOriginal)
        val ivOverlay = cardView.findViewById<ImageView>(R.id.ivOverlay)

        tvName.text = modelName.uppercase()
        tvScore.text = "Score: ${result.score}"
        tvLabel.text = result.label.uppercase()
        tvLabel.setTextColor(getColorForLabel(result.label))

        decodeAndBind(result.images.original, ivOrig)
        decodeAndBind(result.images.overlay, ivOverlay)

        binding.llResultsContainer.addView(cardView)
    }

    private fun getColorForLabel(label: String): Int {
        return if (label == "tumor") resources.getColor(android.R.color.holo_red_light, theme)
        else resources.getColor(android.R.color.holo_green_light, theme)
    }

    private fun decodeAndBind(base64Str: String, targetView: ImageView) {
        try {
            val bytes = Base64.decode(base64Str, Base64.DEFAULT)
            if (bytes.isNotEmpty()) {
                // Glide natively parses GIF bytes out-of-the-box perfectly into loops!
                Glide.with(this).asGif().load(bytes).into(targetView).clearOnDetach()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}