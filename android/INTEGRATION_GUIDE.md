# Android Integration Guide
# Liver Anomaly Detection App

## Architecture Overview

```
Android App  ◄──── HTTP/REST ────►  FastAPI Server  ◄──── PyTorch models
  (Kotlin)           JSON              (Python)            (6 architectures)
```

---

## 1. Prerequisites

### Server side
```bash
pip install fastapi uvicorn python-multipart torch torchvision \
            pydicom scikit-image numpy matplotlib pillow einops kagglehub

# Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Android side
- Android Studio Hedgehog or later
- minSdk 24 (Android 7.0)
- Add to `build.gradle`:

```groovy
dependencies {
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:okhttp:4.11.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
    implementation 'com.github.bumptech.glide:glide:4.16.0'
}
```

---

## 2. Kotlin – API Service Interface

```kotlin
// ApiService.kt
interface ApiService {

    @GET("models")
    suspend fun listModels(): ModelsResponse

    @Multipart
    @POST("predict")
    suspend fun predict(
        @Part file: MultipartBody.Part,
        @Part("model_name") modelName: RequestBody,
        @Part("img_size")   imgSize: RequestBody
    ): PredictResponse

    @Multipart
    @POST("predict/compare")
    suspend fun compareModels(
        @Part file: MultipartBody.Part
    ): CompareResponse
}
```

---

## 3. Kotlin – Data Classes

```kotlin
// Models.kt
data class PredictResponse(
    val score: Double,
    val label: String,           // "normal" or "tumor"
    val model: String,
    val threshold: Double,
    val images: ImageSet,
    val dicom_metadata: DicomMeta
)

data class ImageSet(
    val original: String,        // base64 PNG
    val preprocessed: String,
    val reconstruction: String,
    val error_map: String,
    val overlay: String
)

data class DicomMeta(
    val patient_id: String,
    val modality: String,
    val slice_loc: Double?,
    val pixel_spacing: List<Double>?
)

data class ModelsResponse(val models: List<ModelInfo>)
data class ModelInfo(
    val name: String,
    val description: String,
    val trained: Boolean,
    val threshold: Double
)

data class CompareResponse(
    val majority_vote: String,
    val model_results: Map<String, ModelVote>
)
data class ModelVote(val score: Double, val label: String)
```

---

## 4. Kotlin – Retrofit Client

```kotlin
// RetrofitClient.kt
object RetrofitClient {
    private const val BASE_URL = "http://YOUR_SERVER_IP:8000/"

    private val okhttp = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)   // inference can take ~10s
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        })
        .build()

    val api: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okhttp)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}
```

---

## 5. Kotlin – ViewModel

```kotlin
// AnomalyViewModel.kt
class AnomalyViewModel : ViewModel() {

    val result     = MutableLiveData<PredictResponse?>()
    val isLoading  = MutableLiveData<Boolean>(false)
    val errorMsg   = MutableLiveData<String?>()

    fun predict(context: Context, uri: Uri, modelName: String = "ae_flow") {
        viewModelScope.launch {
            isLoading.value = true
            errorMsg.value  = null
            try {
                val bytes = context.contentResolver
                    .openInputStream(uri)?.readBytes()
                    ?: throw Exception("Cannot read file")

                val body = bytes.toRequestBody("application/octet-stream".toMediaType())
                val part = MultipartBody.Part.createFormData("file", "slice.dcm", body)
                val modelPart = modelName.toRequestBody("text/plain".toMediaType())
                val sizePart  = "256".toRequestBody("text/plain".toMediaType())

                val response = RetrofitClient.api.predict(part, modelPart, sizePart)
                result.value = response
            } catch (e: Exception) {
                errorMsg.value = e.message
            } finally {
                isLoading.value = false
            }
        }
    }
}
```

---

## 6. Kotlin – Activity (Main UI)

```kotlin
// MainActivity.kt
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: AnomalyViewModel by viewModels()

    private val pickDicom = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { uploadAndPredict(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnUpload.setOnClickListener {
            pickDicom.launch("*/*")   // DICOM = application/octet-stream
        }

        viewModel.isLoading.observe(this) { loading ->
            binding.progressBar.isVisible = loading
            binding.btnUpload.isEnabled   = !loading
        }

        viewModel.result.observe(this) { result ->
            result?.let { displayResult(it) }
        }

        viewModel.errorMsg.observe(this) { msg ->
            msg?.let { Toast.makeText(this, it, Toast.LENGTH_LONG).show() }
        }
    }

    private fun uploadAndPredict(uri: Uri) {
        val selectedModel = binding.modelSpinner.selectedItem.toString()
        viewModel.predict(this, uri, selectedModel)
    }

    private fun displayResult(result: PredictResponse) {
        // Classification badge
        val labelColor = if (result.label == "tumor") Color.RED else Color.parseColor("#2E7D32")
        binding.tvLabel.text = result.label.uppercase()
        binding.tvLabel.setTextColor(labelColor)
        binding.tvScore.text = "Score: %.5f".format(result.score)

        // Decode base64 images
        fun b64ToBitmap(b64: String): Bitmap {
            val bytes = Base64.decode(b64, Base64.DEFAULT)
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }

        binding.imgOriginal.setImageBitmap(b64ToBitmap(result.images.original))
        binding.imgPreprocessed.setImageBitmap(b64ToBitmap(result.images.preprocessed))
        binding.imgReconstruction.setImageBitmap(b64ToBitmap(result.images.reconstruction))
        binding.imgErrorMap.setImageBitmap(b64ToBitmap(result.images.error_map))
        binding.imgOverlay.setImageBitmap(b64ToBitmap(result.images.overlay))

        // DICOM metadata
        binding.tvPatientId.text  = "Patient: ${result.dicom_metadata.patient_id}"
        binding.tvModality.text   = "Modality: ${result.dicom_metadata.modality}"
        binding.tvSliceLoc.text   = "Slice Z: ${result.dicom_metadata.slice_loc ?: "N/A"}"
    }
}
```

---

## 7. XML Layout (activity_main.xml)

```xml
<?xml version="1.0" encoding="utf-8"?>
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent" android:layout_height="match_parent"
    android:background="#121212">

  <LinearLayout android:layout_width="match_parent"
      android:layout_height="wrap_content"
      android:orientation="vertical" android:padding="16dp">

    <!-- Header -->
    <TextView android:text="Liver Anomaly Detector"
        android:textSize="22sp" android:textColor="#FFFFFF"
        android:textStyle="bold" android:gravity="center"
        android:layout_width="match_parent" android:layout_height="wrap_content"
        android:paddingBottom="16dp"/>

    <!-- Model selector -->
    <Spinner android:id="@+id/modelSpinner"
        android:layout_width="match_parent" android:layout_height="48dp"
        android:background="#1E1E1E" android:paddingStart="8dp"/>

    <!-- Upload button -->
    <Button android:id="@+id/btnUpload"
        android:text="📂 Select DICOM File"
        android:layout_width="match_parent" android:layout_height="56dp"
        android:layout_marginTop="12dp"
        android:backgroundTint="#1565C0"/>

    <!-- Progress -->
    <ProgressBar android:id="@+id/progressBar"
        android:visibility="gone"
        android:layout_width="wrap_content" android:layout_height="wrap_content"
        android:layout_gravity="center"/>

    <!-- Classification Result -->
    <TextView android:id="@+id/tvLabel"
        android:textSize="32sp" android:textStyle="bold"
        android:gravity="center" android:paddingVertical="12dp"
        android:layout_width="match_parent" android:layout_height="wrap_content"/>
    <TextView android:id="@+id/tvScore"
        android:textColor="#AAAAAA" android:gravity="center"
        android:layout_width="match_parent" android:layout_height="wrap_content"/>

    <!-- 5-panel image grid -->
    <GridLayout android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:columnCount="2" android:rowCount="3"
        android:layout_marginTop="16dp">

      <!-- Each cell: label + ImageView -->
      <!-- (Original) -->
      <LinearLayout android:orientation="vertical"
          android:layout_columnWeight="1" android:layout_rowWeight="1"
          android:layout_width="0dp" android:layout_height="wrap_content"
          android:padding="4dp">
        <TextView android:text="Original" android:textColor="#AAAAAA"
            android:layout_width="wrap_content" android:layout_height="wrap_content"/>
        <ImageView android:id="@+id/imgOriginal"
            android:layout_width="match_parent" android:layout_height="160dp"
            android:scaleType="fitCenter"/>
      </LinearLayout>

      <!-- (Preprocessed HU) -->
      <LinearLayout android:orientation="vertical"
          android:layout_columnWeight="1" android:layout_rowWeight="1"
          android:layout_width="0dp" android:layout_height="wrap_content"
          android:padding="4dp">
        <TextView android:text="Preprocessed (HU)" android:textColor="#AAAAAA"
            android:layout_width="wrap_content" android:layout_height="wrap_content"/>
        <ImageView android:id="@+id/imgPreprocessed"
            android:layout_width="match_parent" android:layout_height="160dp"
            android:scaleType="fitCenter"/>
      </LinearLayout>

      <!-- (Reconstruction) -->
      <LinearLayout android:orientation="vertical"
          android:layout_columnWeight="1" android:layout_rowWeight="1"
          android:layout_width="0dp" android:layout_height="wrap_content"
          android:padding="4dp">
        <TextView android:text="AE Reconstruction" android:textColor="#AAAAAA"
            android:layout_width="wrap_content" android:layout_height="wrap_content"/>
        <ImageView android:id="@+id/imgReconstruction"
            android:layout_width="match_parent" android:layout_height="160dp"
            android:scaleType="fitCenter"/>
      </LinearLayout>

      <!-- (Error Map) -->
      <LinearLayout android:orientation="vertical"
          android:layout_columnWeight="1" android:layout_rowWeight="1"
          android:layout_width="0dp" android:layout_height="wrap_content"
          android:padding="4dp">
        <TextView android:text="Error Heatmap" android:textColor="#AAAAAA"
            android:layout_width="wrap_content" android:layout_height="wrap_content"/>
        <ImageView android:id="@+id/imgErrorMap"
            android:layout_width="match_parent" android:layout_height="160dp"
            android:scaleType="fitCenter"/>
      </LinearLayout>

      <!-- (Overlay – full width) -->
      <LinearLayout android:orientation="vertical"
          android:layout_columnSpan="2"
          android:layout_width="match_parent" android:layout_height="wrap_content"
          android:padding="4dp">
        <TextView android:text="Detected Region Overlay" android:textColor="#AAAAAA"
            android:layout_width="wrap_content" android:layout_height="wrap_content"/>
        <ImageView android:id="@+id/imgOverlay"
            android:layout_width="match_parent" android:layout_height="200dp"
            android:scaleType="fitCenter"/>
      </LinearLayout>

    </GridLayout>

    <!-- Metadata -->
    <LinearLayout android:orientation="vertical"
        android:background="#1E1E1E" android:padding="12dp"
        android:layout_marginTop="12dp" android:layout_borderRadius="8dp"
        android:layout_width="match_parent" android:layout_height="wrap_content">
      <TextView android:id="@+id/tvPatientId" android:textColor="#CCCCCC"
          android:layout_width="wrap_content" android:layout_height="wrap_content"/>
      <TextView android:id="@+id/tvModality"  android:textColor="#CCCCCC"
          android:layout_width="wrap_content" android:layout_height="wrap_content"/>
      <TextView android:id="@+id/tvSliceLoc"  android:textColor="#CCCCCC"
          android:layout_width="wrap_content" android:layout_height="wrap_content"/>
    </LinearLayout>

  </LinearLayout>
</ScrollView>
```

---

## 8. AndroidManifest.xml additions

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>

<!-- If server is plain HTTP (not HTTPS): -->
<application android:usesCleartextTraffic="true" ...>
```

---

## 9. Deployment Options

| Option | How | Best for |
|--------|-----|----------|
| **Local server** | Phone + laptop on same WiFi, use laptop IP | Development / demo |
| **Ngrok tunnel** | `ngrok http 8000` → public URL | Remote testing |
| **Cloud VM** | AWS EC2 / GCP / Azure | Production |
| **Docker** | `docker build + push` | Scalable deployment |

### Docker quick start
```bash
# Build
docker build -t liver-anomaly-api .

# Run (with GPU if available)
docker run -p 8000:8000 --gpus all \
  -v ./checkpoints:/app/checkpoints \
  liver-anomaly-api
```

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 10. API Quick-Test (curl)

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Predict single DICOM
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/slice.dcm" \
  -F "model_name=ae_flow" | python -m json.tool

# Compare all models
curl -X POST http://localhost:8000/predict/compare \
  -F "file=@/path/to/slice.dcm" | python -m json.tool
```
