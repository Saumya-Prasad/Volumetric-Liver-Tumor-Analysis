package com.example.livertumorscanner

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream

class AnomalyViewModel(application: Application) : AndroidViewModel(application) {
    val singlePredictionResult = MutableLiveData<PredictionResponse?>()
    val comparePredictionResult = MutableLiveData<CompareResponse?>()
    val errorState = MutableLiveData<String>()
    val isLoading = MutableLiveData<Boolean>()

    fun uploadDicomFile(uri: Uri, modelName: String) {
        isLoading.value = true
        // Reset old data
        singlePredictionResult.value = null
        comparePredictionResult.value = null

        viewModelScope.launch {
            try {
                val context = getApplication<Application>()
                val inputStream = context.contentResolver.openInputStream(uri)
                // Use .zip extension to allow FastAPI to detect it properly if it's a zip!
                val isZip = context.contentResolver.getType(uri)?.contains("zip") == true || uri.path?.endsWith(".zip") == true
                val fileName = if(isZip) "temp_upload.zip" else "temp_upload.dcm"
                
                val tempFile = File(context.cacheDir, fileName)
                val out = FileOutputStream(tempFile)
                inputStream?.copyTo(out)
                out.close()
                inputStream?.close()

                val requestFile = tempFile.asRequestBody("application/octet-stream".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData("file", tempFile.name, requestFile)
                val sizeBody = "256".toRequestBody("text/plain".toMediaTypeOrNull())

                if (modelName == "ALL MODELS") {
                    val response = RetrofitClient.api.predictCompare(body, sizeBody)
                    if (response.isSuccessful && response.body() != null) {
                        comparePredictionResult.postValue(response.body())
                    } else {
                        errorState.postValue("Server Error: ${response.code()}")
                    }
                } else {
                    val modelBody = modelName.toRequestBody("text/plain".toMediaTypeOrNull())
                    val response = RetrofitClient.api.predictSliceOrZip(body, modelBody, sizeBody)
                    if (response.isSuccessful && response.body() != null) {
                        singlePredictionResult.postValue(response.body())
                    } else {
                        errorState.postValue("Server Error: ${response.code()}")
                    }
                }

            } catch (e: Exception) {
                errorState.postValue("Failed to connect: ${e.message}")
            } finally {
                isLoading.postValue(false)
            }
        }
    }
}
