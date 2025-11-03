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
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.chiron.R
import com.example.chiron.ui.theme.ChironTheme
import com.example.chiron.ui.theme.ExplosivePurple

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
                    SkullWithPrediction(
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
    var lastPredictionDate by remember { mutableStateOf<String?>(null) }

    // Helper function to load/refresh biological age prediction
    fun loadBiologicalAge() {
        coroutineScope.launch(Dispatchers.IO) {
            try {
                isLoadingAge = true
                // Use previous days' files (not today) for prediction
                var result = CosinorAgePredictor.predictCosinorAgeFromPreviousDays(context, age = 39, gender = "male")
                
                // If no previous files, try using test data
                if (!result.success) {
                    val testFile = CosinorAgePredictor.copyTestFileFromAssets(context, "accelerometer_20251030.csv")
                    if (testFile != null && testFile.exists()) {
                        result = CosinorAgePredictor.predictCosinorAgeFromFile(context, testFile, age = 39, gender = "male")
                        if (result.success) {
                            ageMessage = "Using test data"
                        }
                    }
                }

                withContext(Dispatchers.Main) {
                    if (result.success && result.cosinorAge != null) {
                        biologicalAge = result.cosinorAge
                        ageMessage = ""
                        // Update last prediction date
                        val dateFormat = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.getDefault())
                        lastPredictionDate = dateFormat.format(java.util.Date())
                    } else {
                        ageMessage = result.message
                    }
                    isLoadingAge = false
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    ageMessage = "Error: ${e.message}"
                    isLoadingAge = false
                }
            }
        }
    }

    // Load biological age on composition (using previous files)
    LaunchedEffect(Unit) {
        loadBiologicalAge()
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
                    
                    // Check if a new day has started and refresh prediction if needed
                    val dateFormat = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.getDefault())
                    val currentDate = dateFormat.format(java.util.Date())
                    var shouldRefresh = false
                    
                    withContext(Dispatchers.Main) {
                        if (lastPredictionDate != null && lastPredictionDate != currentDate) {
                            // New day detected - refresh prediction
                            shouldRefresh = true
                        } else if (lastPredictionDate == null) {
                            // First time - load prediction
                            shouldRefresh = true
                        }
                    }
                    
                    if (shouldRefresh && !isLoadingAge) {
                        loadBiologicalAge()
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
            text = "LIVE LONG AND PROSPER",
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
                Text(
                    text = "Biological Age: ${String.format("%.1f", biologicalAge)} years",
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

@Preview(showBackground = true)
@Composable
fun ImageViewPreview() {
    ChironTheme {
        ImageView()
    }
}