package com.example.livertumorscanner

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {
    @Multipart
    @POST("/predict")
    suspend fun predictSliceOrZip(
        @Part file: MultipartBody.Part,
        @Part("model_name") modelName: RequestBody,
        @Part("img_size") imgSize: RequestBody
    ): Response<PredictionResponse>

    @Multipart
    @POST("/predict/compare")
    suspend fun predictCompare(
        @Part file: MultipartBody.Part,
        @Part("img_size") imgSize: RequestBody
    ): Response<CompareResponse>
}
