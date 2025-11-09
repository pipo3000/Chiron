package com.example.chiron

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.State
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import androidx.compose.runtime.rememberCoroutineScope
import androidx.lifecycle.lifecycleScope
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.chiron.R
import com.example.chiron.ui.theme.ChironTheme
import com.example.chiron.ui.theme.ExplosivePurple
import java.util.Locale

class MainActivity : ComponentActivity() {
    private var accelerometerService: AccelerometerDataService? = null
    private var isServiceBound = false
    private val isRecordingState = mutableStateOf(false)
    private val recordCountState = mutableStateOf(0)

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as AccelerometerDataService.LocalBinder
            accelerometerService = binder.getService()
            isServiceBound = true
            
            // Update state immediately
            updateServiceState()
            
            // Also update after a delay to ensure recording has started
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                updateServiceState()
            }, 500)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            accelerometerService = null
            isServiceBound = false
            isRecordingState.value = false
            recordCountState.value = 0
        }
    }

    private fun updateServiceState() {
        accelerometerService?.let {
            isRecordingState.value = it.isRecording()
            recordCountState.value = it.getRecordCount()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Request notification permission for Android 13+
        requestNotificationPermission()
        
        // Request storage permission for accessing Downloads directory
        requestStoragePermission()
        
        // Generate synthetic data files if none exist (on first install)
        lifecycleScope.launch(Dispatchers.IO) {
            val generated = CosinorAgePredictor.generateSyntheticDataIfNeeded(this@MainActivity)
            if (generated > 0) {
                android.util.Log.d("MainActivity", "Generated $generated synthetic data files on first launch")
            }
        }
        
        // Start the background service automatically for continuous recording
        // Use post to ensure UI is set up first
        android.os.Handler(android.os.Looper.getMainLooper()).post {
            startContinuousRecording()
        }
        
        setContent {
            ChironTheme(
                dynamicColor = false // Disable dynamic colors to use custom background
            ) {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(
                        isRecording = isRecordingState.value,
                        onUpdateRecordingState = { updateServiceState() }
                    )
                }
            }
        }
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    100
                )
            }
        }
    }
    
    private fun requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Android 11+ requires MANAGE_EXTERNAL_STORAGE for accessing Downloads
            if (!android.os.Environment.isExternalStorageManager()) {
                try {
                    val intent = android.content.Intent(android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                    intent.addCategory("android.intent.category.DEFAULT")
                    intent.data = android.net.Uri.parse(String.format("package:%s", packageName))
                    startActivity(intent)
                } catch (e: Exception) {
                    // Fallback to general storage settings
                    val intent = android.content.Intent(android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                    startActivity(intent)
                }
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // Android 6-10: Request WRITE_EXTERNAL_STORAGE
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                    200
                )
            }
        }
    }

    private fun startContinuousRecording() {
        // Start the background service automatically for continuous recording
        val serviceIntent = Intent(this, AccelerometerDataService::class.java).apply {
            action = AccelerometerDataService.ACTION_START
        }
        
        // Check if service is already running to avoid duplicate starts
        // This prevents "Operation not started" errors from AppOps
        if (!isServiceBound) {
            // Start the service first
            try {
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent)
                } else {
                    startService(serviceIntent)
                }
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Failed to start service: ${e.message}", e)
                return
            }
            
            // Bind to service to get status updates
            try {
                bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Failed to bind service: ${e.message}", e)
            }
            
            // Update state multiple times to ensure we get the correct status
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                updateServiceState()
            }, 1000)
            
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                updateServiceState()
            }, 2000)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isServiceBound) {
            unbindService(serviceConnection)
            isServiceBound = false
        }
    }
}

@Composable
fun SkullWithPrediction(
    isRecording: Boolean = false,
    onUpdateRecordingState: () -> Unit = {}
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var fileCount by remember { mutableStateOf(0) }
    var biologicalAge by remember { mutableStateOf<Double?>(null) }
    var isLoadingAge by remember { mutableStateOf(false) }
    var ageMessage by remember { mutableStateOf("") }
    
    // Load last prediction date and file hash from SharedPreferences (persistent storage)
    val prefs = remember { context.getSharedPreferences("ChironPrefs", android.content.Context.MODE_PRIVATE) }
    
    // Clear prediction cache on first load after code update (to force recalculation with ENMO fixes)
    // This ensures old predictions with buggy ENMO values get recalculated
    val predictionCacheVersion = prefs.getInt("prediction_cache_version", 0)
    val CURRENT_CACHE_VERSION = 2  // Increment when ENMO calculation changes
    if (predictionCacheVersion < CURRENT_CACHE_VERSION) {
        prefs.edit().apply {
            remove("last_prediction_date")
            remove("last_prediction_file_hash")
            putInt("prediction_cache_version", CURRENT_CACHE_VERSION)
            apply()
        }
        android.util.Log.d("MainActivity", "Cleared prediction cache due to version update (ENMO fix)")
    }
    
    var lastPredictionDate by remember { 
        mutableStateOf(prefs.getString("last_prediction_date", null))
    }
    var lastPredictionFileHash by remember {
        mutableStateOf(prefs.getString("last_prediction_file_hash", null))
    }

    // Helper function to compute hash of files used for prediction
    suspend fun computePredictionFileHash(): String {
        return withContext(Dispatchers.IO) {
            val files = CosinorAgePredictor.getAllPreviousDayFiles(context, maxDays = 7)
            // Create a hash based on file names, sizes, and modification times
            val hashString = files.sortedBy { it.name }.joinToString("|") { 
                "${it.name}:${it.length()}:${it.lastModified()}"
            }
            // Simple hash (can use proper hash function if needed)
            hashString.hashCode().toString()
        }
    }

    // Helper function to save last prediction info persistently
    fun saveLastPredictionInfo(date: String, fileHash: String) {
        prefs.edit().apply {
            putString("last_prediction_date", date)
            putString("last_prediction_file_hash", fileHash)
            apply()
        }
        lastPredictionDate = date
        lastPredictionFileHash = fileHash
    }

    // Helper function to check if we need to update prediction
    // Updates if: date changed, files changed, or never predicted
    // NOTE: Always recalculates on app startup to ensure ENMO corrections are applied
    suspend fun shouldUpdatePrediction(forceRecalculate: Boolean = false): Boolean {
        return withContext(Dispatchers.IO) {
            val dateFormat = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.getDefault())
            val currentDate = dateFormat.format(java.util.Date())
            
            // Check if we have files to use
            val files = CosinorAgePredictor.getAllPreviousDayFiles(context, maxDays = 7)
            if (files.isEmpty()) {
                return@withContext false
            }
            
            // Compute current file hash
            val currentFileHash = computePredictionFileHash()
            
            // Update if:
            // 1. Force recalculate (e.g., on app startup to apply ENMO corrections)
            // 2. Never predicted
            // 3. Date changed (new day with potentially new complete data)
            // 4. Files changed (new files added, files modified)
            val needsUpdate = forceRecalculate ||
                             lastPredictionDate == null || 
                             lastPredictionDate != currentDate || 
                             lastPredictionFileHash != currentFileHash
            
            if (needsUpdate) {
                android.util.Log.d("MainActivity", "Prediction update needed: forceRecalculate=$forceRecalculate, dateChanged=${lastPredictionDate != currentDate}, filesChanged=${lastPredictionFileHash != currentFileHash}, currentFiles=${files.size}")
            }
            
            return@withContext needsUpdate
        }
    }

    // Helper function to load/refresh biological age prediction
    fun loadBiologicalAge() {
        if (isLoadingAge) return // Prevent concurrent loads
        
        coroutineScope.launch(Dispatchers.IO) {
            try {
                isLoadingAge = true
                val fileHash = computePredictionFileHash()

                val result = CosinorAgePredictor.predictCosinorAgeFromPreviousDays(
                    context = context,
                    age = 39,
                    gender = "male",
                    maxDays = 7
                )

                val dateFormat = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.getDefault())
                val currentDate = dateFormat.format(java.util.Date())

                withContext(Dispatchers.Main) {
                    if (result.success && result.cosinorAge != null) {
                        val oldBiologicalAge = biologicalAge
                        biologicalAge = result.cosinorAge
                        ageMessage = ""

                        if (oldBiologicalAge != null && oldBiologicalAge != result.cosinorAge) {
                            android.util.Log.d("MainActivity", "Biological age updated: ${oldBiologicalAge} -> ${result.cosinorAge}")
                        } else if (oldBiologicalAge != null && oldBiologicalAge == result.cosinorAge) {
                            android.util.Log.d("MainActivity", "Biological age unchanged: ${result.cosinorAge}")
                        }

                        saveLastPredictionInfo(currentDate, fileHash)
                    } else {
                        biologicalAge = null
                        ageMessage = result.message.ifBlank { "Prediction unavailable" }
                    }
                    isLoadingAge = false
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    ageMessage = "Error: ${e.message}"
                    biologicalAge = null
                    isLoadingAge = false
                }
            }
        }
    }

    // Load biological age on composition (using previous files)
    LaunchedEffect(Unit) {
        // Always load prediction on startup to display it
        // Force recalculation on startup to ensure ENMO corrections are applied
        coroutineScope.launch(Dispatchers.IO) {
            // Always recalculate on startup to ensure:
            // 1. ENMO corrections from old buggy files are applied
            // 2. Latest cosinor parameters are computed
            // 3. Prediction reflects current data state
            android.util.Log.d("MainActivity", "Loading biological age prediction on startup")
            loadBiologicalAge()
        }
    }

    // Update recording state and file count periodically, and check for new day
    LaunchedEffect(Unit) {
        while (true) {
            coroutineScope.launch(Dispatchers.IO) {
                try {
                    // Update recording state
                    withContext(Dispatchers.Main) {
                        onUpdateRecordingState()
                    }
                    
                    // Check if a new day has started with complete data and refresh prediction if needed
                    val shouldUpdate = shouldUpdatePrediction()
                    withContext(Dispatchers.Main) {
                        if (shouldUpdate && !isLoadingAge) {
                            loadBiologicalAge()
                        }
                    }
                    
                    // Count recorded files - check both Downloads and app storage
                    var files: Array<java.io.File>? = null
                    val downloadsDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS)
                    val chironDir = java.io.File(downloadsDir, "Chiron")
                    if (chironDir.exists() && chironDir.isDirectory) {
                        files = chironDir.listFiles { _, name ->
                            name.startsWith("accelerometer_") && name.endsWith(".csv")
                        }
                    }
                    
                    // Also check app's external files directory
                    if (files == null || files.isEmpty()) {
                        val appDir = context.getExternalFilesDir(null) ?: context.filesDir
                        val appChironDir = java.io.File(appDir, "Chiron")
                        if (appChironDir.exists() && appChironDir.isDirectory) {
                            files = appChironDir.listFiles { _, name ->
                                name.startsWith("accelerometer_") && name.endsWith(".csv")
                            }
                        }
                    }
                    
                    withContext(Dispatchers.Main) {
                        fileCount = files?.size ?: 0
                    }
                } catch (e: Exception) {
                    // Ignore errors
                }
            }
            delay(2000) // Update every 2 seconds
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title above the image
        Text(
            text = "YOU ARE ONLY YOUNG, BUT YOU ARE GONNA DIE!",
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier
                .padding(top = 48.dp, bottom = 64.dp)
        )
        
        // Image view - takes most of the space
        ImageView(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        )
        
        // Biological age and status display below the image
        Column(
            modifier = Modifier.padding(top = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Biological age display in explosive purple
            if (isLoadingAge) {
                Text(
                    text = "Calculating...",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            } else if (biologicalAge != null) {
                val formattedAge = String.format(Locale.US, "%.1f", biologicalAge!!)
                Text(
                    text = "Biological Age (last 7 days): $formattedAge years",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    color = ExplosivePurple,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
            } else if (ageMessage.isNotEmpty()) {
                Text(
                    text = ageMessage,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }
            
            // Recording status indicator
            Text(
                text = if (isRecording) "● Recording" else "○ Not Recording",
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                color = if (isRecording) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            // File count display
            Text(
                text = "Recorded files: $fileCount",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun ImageView(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier,
        contentAlignment = Alignment.Center
    ) {
        // To use your own image:
        // 1. Place your image (PNG, JPG, etc.) in app/src/main/res/drawable/
        // 2. Replace "ic_launcher_background" below with your image filename (without extension)
        //    Example: If your file is "my_picture.png", use R.drawable.my_picture
        Image(
            painter = painterResource(id = R.drawable.skull), // Change this to your image resource
            contentDescription = "Display Image",
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .aspectRatio(1f), // Change to aspectRatio(16f / 9f) for landscape, or remove for auto
            contentScale = ContentScale.Fit // Use ContentScale.Crop for fill, ContentScale.Fit for fit
        )
    }
}

@Composable
fun MainScreen(
    isRecording: Boolean = false,
    onUpdateRecordingState: () -> Unit = {}
) {
    var selectedTabIndex by remember { mutableStateOf(0) }
    val tabs = listOf("Prediction", "Visualization")
    
    Column(modifier = Modifier.fillMaxSize()) {
        TabRow(selectedTabIndex = selectedTabIndex) {
            tabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTabIndex == index,
                    onClick = { selectedTabIndex = index },
                    text = { Text(title) }
                )
            }
        }
        
        when (selectedTabIndex) {
            0 -> SkullWithPrediction(
                isRecording = isRecording,
                onUpdateRecordingState = onUpdateRecordingState
            )
            1 -> DataVisualizationScreen()
        }
    }
}

@Composable
fun DataVisualizationScreen() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var enmoValues by remember { mutableStateOf<List<Double>>(emptyList()) }
    var cosinorFit by remember { mutableStateOf<List<Double>>(emptyList()) }
    var cosinorParams by remember { mutableStateOf<Map<String, Double>>(emptyMap()) }
    var rhythmRobustness by remember { mutableStateOf<Double?>(null) }
    
    // Load visualization data
    LaunchedEffect(Unit) {
        coroutineScope.launch(Dispatchers.IO) {
            try {
                isLoading = true
                errorMessage = null
                
                // Use the same files as the prediction (last 7 full days, excluding today)
                val files = CosinorAgePredictor.getAllPreviousDayFiles(context, maxDays = 7)
                if (files.isEmpty()) {
                    errorMessage = "No data files available for visualization"
                    isLoading = false
                    return@launch
                }
                
                // Concatenate files
                val tempDir = context.cacheDir
                val concatenatedFile = java.io.File(tempDir, "viz_${System.currentTimeMillis()}.csv")
                
                if (!CosinorAgePredictor.concatenateCsvFiles(files, concatenatedFile)) {
                    errorMessage = "Failed to prepare data for visualization"
                    isLoading = false
                    return@launch
                }
                
                // Get visualization data
                val result = CosinorAgePredictor.getVisualizationData(
                    context,
                    concatenatedFile,
                    age = 39,
                    gender = "male"
                )
                
                // Clean up temp file
                try {
                    if (concatenatedFile.exists()) {
                        concatenatedFile.delete()
                    }
                } catch (e: Exception) {
                    // Ignore cleanup errors
                }
                
                withContext(Dispatchers.Main) {
                    if (result.success) {
                        enmoValues = result.enmoValues
                        cosinorFit = result.cosinorFit
                        cosinorParams = result.cosinorParams
                        rhythmRobustness = result.rhythmRobustness
                    } else {
                        errorMessage = result.message
                    }
                    isLoading = false
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    errorMessage = "Error: ${e.message}"
                    isLoading = false
                }
            }
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "ENMO & Cosinor Fit Visualization",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        
        if (isLoading) {
            Text(
                text = "Loading data...",
                modifier = Modifier.padding(32.dp)
            )
        } else if (errorMessage != null) {
            Text(
                text = errorMessage!!,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(16.dp)
            )
        } else if (enmoValues.isEmpty()) {
            Text(
                text = "No data available",
                modifier = Modifier.padding(32.dp)
            )
        } else {
            // Display cosinor parameters
            if (cosinorParams.isNotEmpty()) {
                Column(
                    modifier = Modifier.padding(bottom = 16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    cosinorParams["mesor"]?.let {
                        Text("Mesor: ${String.format("%.4f", it)}", modifier = Modifier.padding(4.dp))
                    }
                    cosinorParams["amplitude"]?.let {
                        Text("Amplitude: ${String.format("%.4f", it)}", modifier = Modifier.padding(4.dp))
                    }
                    cosinorParams["acrophase"]?.let {
                        Text("Acrophase: ${String.format("%.4f", it)}", modifier = Modifier.padding(4.dp))
                    }
                    rhythmRobustness?.let {
                        Text(
                            text = "Rhythm Robustness (R²): ${String.format("%.3f", it)}",
                            modifier = Modifier.padding(4.dp)
                        )
                    }
                }
            }
            
            // Chart visualization
            ENMOChart(
                enmoValues = enmoValues,
                cosinorFit = cosinorFit,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )
        }
    }
}

@Composable
fun ENMOChart(
    enmoValues: List<Double>,
    cosinorFit: List<Double>,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // Chart area
        Box(modifier = Modifier.weight(1f)) {
            Canvas(modifier = Modifier.fillMaxSize()) {
            if (enmoValues.isEmpty()) return@Canvas
            
            val padding = 50.dp.toPx()  // Increased padding for axis labels
            val chartWidth = size.width - 2 * padding
            val chartHeight = size.height - 2 * padding
            
            // Find min/max for scaling
            val allValues = (enmoValues + cosinorFit).filter { it.isFinite() }
            val minY = allValues.minOrNull() ?: 0.0
            val maxY = allValues.maxOrNull() ?: 1.0
            val yRange = (maxY - minY).coerceAtLeast(0.001)
            
            // Draw axes
            drawLine(
                color = androidx.compose.ui.graphics.Color.Gray,
                start = androidx.compose.ui.geometry.Offset(padding, padding),
                end = androidx.compose.ui.geometry.Offset(padding, size.height - padding),
                strokeWidth = 2.dp.toPx()
            )
            drawLine(
                color = androidx.compose.ui.graphics.Color.Gray,
                start = androidx.compose.ui.geometry.Offset(padding, size.height - padding),
                end = androidx.compose.ui.geometry.Offset(size.width - padding, size.height - padding),
                strokeWidth = 2.dp.toPx()
            )
            
            // Draw ENMO data
            if (enmoValues.isNotEmpty()) {
                val stepX = chartWidth / (enmoValues.size - 1).coerceAtLeast(1)
                val path = androidx.compose.ui.graphics.Path()
                
                enmoValues.forEachIndexed { index, value ->
                    val x = padding + index * stepX
                    val y = size.height - padding - ((value - minY).toFloat() / yRange.toFloat() * chartHeight)
                    
                    if (index == 0) {
                        path.moveTo(x, y)
                    } else {
                        path.lineTo(x, y)
                    }
                }
                
                drawPath(
                    path = path,
                    color = androidx.compose.ui.graphics.Color.Blue,
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2.dp.toPx())
                )
            }
            
            // Draw cosinor fit
            if (cosinorFit.isNotEmpty() && cosinorFit.size == enmoValues.size) {
                val stepX = chartWidth / (cosinorFit.size - 1).coerceAtLeast(1)
                val path = androidx.compose.ui.graphics.Path()
                
                cosinorFit.forEachIndexed { index, value ->
                    val x = padding + index * stepX
                    val y = size.height - padding - ((value - minY).toFloat() / yRange.toFloat() * chartHeight)
                    
                    if (index == 0) {
                        path.moveTo(x, y)
                    } else {
                        path.lineTo(x, y)
                    }
                }
                
                drawPath(
                    path = path,
                    color = androidx.compose.ui.graphics.Color.Red,
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2.dp.toPx())
                )
            }
            
            // Draw Y-axis label (rotated) - positioned on the left side
            val yAxisPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.GRAY
                textSize = 12f * density
                textAlign = android.graphics.Paint.Align.CENTER
            }
            val nativeCanvas = drawContext.canvas.nativeCanvas
            nativeCanvas.save()
            nativeCanvas.translate(15.dp.toPx(), size.height / 2)
            nativeCanvas.rotate(-90f)
            nativeCanvas.drawText("ENMO (mg)", 0f, 0f, yAxisPaint)
            nativeCanvas.restore()
            
            // Draw X-axis label - positioned at the bottom center
            val xAxisPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.GRAY
                textSize = 12f * density
                textAlign = android.graphics.Paint.Align.CENTER
            }
            drawContext.canvas.nativeCanvas.drawText(
                "Time (minutes)",
                size.width / 2 - 50.dp.toPx(),
                size.height - 10.dp.toPx(),
                xAxisPaint
            )
            }
        }
        
        // Legend - now below the chart
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = androidx.compose.foundation.layout.Arrangement.Center
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(end = 16.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(16.dp)
                        .background(androidx.compose.ui.graphics.Color.Blue)
                )
                Text(" ENMO Data", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(start = 4.dp))
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(16.dp)
                        .background(androidx.compose.ui.graphics.Color.Red)
                )
                Text(" Cosinor Fit", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(start = 4.dp))
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ImageViewPreview() {
    ChironTheme {
        ImageView()
    }
}