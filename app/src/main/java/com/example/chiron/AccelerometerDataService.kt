package com.example.chiron

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Binder
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class AccelerometerDataService : Service() {
    private val binder = LocalBinder()
    private var sensorManager: SensorManager? = null
    private var accelerometer: Sensor? = null
    private var sensorListener: SensorEventListener? = null
    private var isRecording = false
    private var dataFile: File? = null
    private var fileWriter: FileWriter? = null
    private var currentDay: String = ""
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // Wake lock to keep device awake for continuous recording
    private var wakeLock: PowerManager.WakeLock? = null
    
    // Statistics
    private var recordCount = 0
    private var startTime: Long = 0
    
    // Minute-level aggregation for CosinorAge format
    private data class MinuteData(
        val minuteTimestamp: Long, // Start of minute in milliseconds
        val enmoValues: MutableList<Double> = mutableListOf()
    )
    private var currentMinuteData: MinuteData? = null
    private val minuteAggregationLock = Any()

    inner class LocalBinder : Binder() {
        fun getService(): AccelerometerDataService = this@AccelerometerDataService
    }

    override fun onBind(intent: Intent): IBinder {
        return binder
    }

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        createNotificationChannel()
        
        // Acquire wake lock to keep device awake for continuous recording
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "Chiron::AccelerometerWakeLock"
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Automatically start recording when service is started
        // This ensures continuous recording even if service is restarted
        if (!isRecording) {
            startRecording()
        }
        
        when (intent?.action) {
            ACTION_START -> startRecording()
            ACTION_STOP -> stopRecording()
        }
        
        // Return START_STICKY to automatically restart service if killed by system
        // Return START_NOT_STICKY if we don't want auto-restart
        return START_STICKY
    }

    fun startRecording() {
        if (isRecording) return
        
        isRecording = true
        recordCount = 0
        startTime = System.currentTimeMillis()
        
        // Start foreground service FIRST (before other operations)
        // This ensures proper AppOps tracking
        try {
            startForeground(NOTIFICATION_ID, createNotification())
        } catch (e: Exception) {
            android.util.Log.e("AccelerometerService", "Failed to start foreground: ${e.message}", e)
        }
        
        // Cleanup old files (older than 10 days)
        serviceScope.launch {
            cleanupOldFiles()
        }
        
        // Create/append to daily file - this is async, but recording can start anyway
        // The writeMinuteData function will wait for fileWriter to be ready
        openOrCreateDailyFile()
        
        sensorListener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                event?.let {
                    // Check if we need to switch to a new day's file
                    checkAndSwitchDailyFile()
                    
                    val x = it.values[0]
                    val y = it.values[1]
                    val z = it.values[2]
                    val timestamp = System.currentTimeMillis()
                    
                    // Calculate ENMO: ENMO = max(0, sqrt(x² + y² + z²) - 1g)
                    // Android accelerometer returns values in m/s², so convert to g units first
                    val GRAVITY_MS2 = 9.80665  // Standard gravity in m/s²
                    val magnitude = kotlin.math.sqrt(x * x + y * y + z * z)
                    val magnitudeG = magnitude / GRAVITY_MS2  // Convert m/s² to g units
                    val enmo = (magnitudeG - 1.0).coerceAtLeast(0.0)  // ENMO in g units
                    
                    recordAccelerometerData(timestamp, x, y, z, enmo)
                    recordCount++
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
                // Handle accuracy changes if needed
            }
        }
        
        // Start periodic minute-level aggregation and writing
        startMinuteAggregation()

        accelerometer?.let {
            sensorManager?.registerListener(
                sensorListener,
                it,
                SensorManager.SENSOR_DELAY_NORMAL
            )
        }

        // Acquire wake lock to prevent device from sleeping
        // Only acquire if not already held to avoid "Operation not started" errors
        if (wakeLock?.isHeld != true) {
            try {
                wakeLock?.acquire(10 * 60 * 60 * 1000L /*10 hours*/)
            } catch (e: Exception) {
                android.util.Log.e("AccelerometerService", "Failed to acquire wake lock: ${e.message}", e)
            }
        }
        
        // Update notification every 2 seconds while recording
        serviceScope.launch {
            while (isRecording) {
                delay(2000)
                if (isRecording) {
                    updateNotification("Recording...")
                    // Check daily file switch periodically
                    checkAndSwitchDailyFile()
                }
            }
        }
    }
    
    private fun openOrCreateDailyFile() {
        val today = SimpleDateFormat("yyyyMMdd", Locale.getDefault()).format(Date())
        currentDay = today
        val fileName = "accelerometer_$today.csv"
        dataFile = getExternalStorageFile(fileName)
        
        serviceScope.launch {
            try {
                // Ensure directory exists
                dataFile?.parentFile?.mkdirs()
                
                // Check if file exists - if not, write header
                val fileExists = dataFile?.exists() == true
                fileWriter = FileWriter(dataFile, true) // Append mode
                
                if (!fileExists) {
                    // Write CSV header for CosinorAge format: timestamp,enmo
                    // Timestamp format: ISO 8601 (YYYY-MM-DDTHH:mm:ss)
                    fileWriter?.write("timestamp,enmo\n")
                    fileWriter?.flush()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                isRecording = false
            }
        }
    }
    
    private fun checkAndSwitchDailyFile() {
        val today = SimpleDateFormat("yyyyMMdd", Locale.getDefault()).format(Date())
        if (currentDay != today) {
            // New day - close old file and open new one
            serviceScope.launch {
                try {
                    fileWriter?.flush()
                    fileWriter?.close()
                } catch (e: Exception) {
                    e.printStackTrace()
                }
                
                // Open new day's file
                openOrCreateDailyFile()
                
                // Cleanup old files periodically (every day)
                cleanupOldFiles()
            }
        }
    }
    
    private fun cleanupOldFiles() {
        try {
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val chironDir = File(downloadsDir, "Chiron")
            
            if (!chironDir.exists() || !chironDir.isDirectory) {
                return
            }
            
            val files = chironDir.listFiles { _, name ->
                name.startsWith("accelerometer_") && name.endsWith(".csv")
            }
            
            if (files == null) return
            
            val calendar = java.util.Calendar.getInstance()
            calendar.add(java.util.Calendar.DAY_OF_YEAR, -10) // 10 days ago
            val cutoffDate = calendar.time
            
            var deletedCount = 0
            for (file in files) {
                // Extract date from filename: accelerometer_YYYYMMDD.csv
                val fileName = file.name
                if (fileName.length >= 23) {
                    val dateStr = fileName.substring(14, 22) // Extract YYYYMMDD
                    try {
                        val fileDate = SimpleDateFormat("yyyyMMdd", Locale.getDefault()).parse(dateStr)
                        if (fileDate != null && fileDate.before(cutoffDate)) {
                            if (file.delete()) {
                                deletedCount++
                            }
                        }
                    } catch (e: Exception) {
                        // Skip files with invalid date format
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun stopRecording() {
        if (!isRecording) return
        
        isRecording = false
        sensorManager?.unregisterListener(sensorListener)
        
        // Release wake lock when stopping
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
        
        serviceScope.launch {
            var dataToWrite: MinuteData? = null
            
            synchronized(minuteAggregationLock) {
                // Prepare any remaining minute data for writing
                currentMinuteData?.let { minuteData ->
                    if (minuteData.enmoValues.isNotEmpty()) {
                        // Copy data to write outside synchronized block
                        dataToWrite = MinuteData(
                            minuteData.minuteTimestamp,
                            ArrayList(minuteData.enmoValues)
                        )
                    }
                }
                currentMinuteData = null
            }
            
            // Write outside synchronized block
            dataToWrite?.let { writeMinuteData(it) }
            
            // Wait a bit for the write to complete
            kotlinx.coroutines.delay(100)
            
            try {
                fileWriter?.flush() // Ensure all data is written
                fileWriter?.close()
                fileWriter = null
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        
        val filePath = dataFile?.absolutePath ?: "unknown"
        val displayPath = if (filePath.contains("Chiron")) {
            "Downloads/Chiron/${dataFile?.name ?: "unknown"}"
        } else {
            dataFile?.name ?: "unknown"
        }
        updateNotification("Recording stopped. Saved to: $displayPath")
        stopForeground(STOP_FOREGROUND_REMOVE)
        // Don't call stopSelf() - let the service keep running
        // Only stop the service if explicitly requested
    }
    
    fun forceStopService() {
        stopRecording()
        stopSelf()
    }

    /**
     * Record accelerometer data and aggregate by minute for CosinorAge format.
     * ENMO (Euclidean Norm Minus One) is calculated and aggregated per minute.
     */
    private fun recordAccelerometerData(timestamp: Long, x: Float, y: Float, z: Float, enmo: Double) {
        var dataToWrite: MinuteData? = null
        
        synchronized(minuteAggregationLock) {
            // Get the start of the current minute (rounded down to minute)
            val minuteStart = (timestamp / 60000) * 60000 // Round down to nearest minute
            
            // Initialize or update current minute data
            if (currentMinuteData == null || currentMinuteData!!.minuteTimestamp != minuteStart) {
                // New minute started - prepare previous minute's data for writing
                currentMinuteData?.let { minuteData ->
                    if (minuteData.enmoValues.isNotEmpty()) {
                        // Copy data to write outside synchronized block
                        dataToWrite = MinuteData(
                            minuteData.minuteTimestamp,
                            ArrayList(minuteData.enmoValues) // Copy the list
                        )
                    }
                }
                // Start new minute
                currentMinuteData = MinuteData(minuteStart)
            }
            
            // Add ENMO value to current minute
            currentMinuteData?.enmoValues?.add(enmo)
        }
        
        // Write data outside synchronized block to avoid blocking
        dataToWrite?.let { writeMinuteData(it) }
    }
    
    /**
     * Write minute-level aggregated data to file in CosinorAge format.
     * Format: timestamp,enmo where timestamp is ISO 8601 format.
     */
    private fun writeMinuteData(minuteData: MinuteData) {
        serviceScope.launch(Dispatchers.IO) {
            try {
                // Wait for file writer to be ready (in case file is still being opened)
                var retries = 10
                while (fileWriter == null && retries > 0 && isRecording) {
                    kotlinx.coroutines.delay(100)
                    retries--
                }
                
                if (fileWriter == null) {
                    android.util.Log.e("AccelerometerService", "FileWriter is null, cannot write data")
                    return@launch
                }
                
                // Calculate mean ENMO for this minute
                val meanEnmo = if (minuteData.enmoValues.isNotEmpty()) {
                    minuteData.enmoValues.average()
                } else {
                    return@launch
                }
                
                // Format timestamp as ISO 8601 (YYYY-MM-DDTHH:mm:ss)
                val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
                val timestampStr = dateFormat.format(Date(minuteData.minuteTimestamp))
                
                // Write in CosinorAge format: timestamp,enmo
                fileWriter?.write("$timestampStr,${String.format(Locale.US, "%.6f", meanEnmo)}\n")
                fileWriter?.flush() // Always flush after writing a minute
            } catch (e: Exception) {
                android.util.Log.e("AccelerometerService", "Error writing minute data: ${e.message}", e)
                e.printStackTrace()
            }
        }
    }
    
    /**
     * Start periodic task to flush minute-level data.
     * This ensures data is written even if we're in the middle of a minute.
     */
    private fun startMinuteAggregation() {
        serviceScope.launch {
            while (isRecording) {
                delay(60000) // Every minute
                var dataToWrite: MinuteData? = null
                
                synchronized(minuteAggregationLock) {
                    // Force write current minute if it has data
                    currentMinuteData?.let { minuteData ->
                        if (minuteData.enmoValues.isNotEmpty()) {
                            // Copy data to write outside synchronized block
                            dataToWrite = MinuteData(
                                minuteData.minuteTimestamp,
                                ArrayList(minuteData.enmoValues) // Copy the list
                            )
                            // Reset for next minute
                            val nextMinute = (System.currentTimeMillis() / 60000) * 60000
                            currentMinuteData = MinuteData(nextMinute)
                        }
                    }
                }
                
                // Write outside synchronized block
                dataToWrite?.let { writeMinuteData(it) }
                
                // Flush file writer
                try {
                    fileWriter?.flush()
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Accelerometer Data Recording",
                NotificationManager.IMPORTANCE_DEFAULT  // Changed from LOW to DEFAULT for visibility
            ).apply {
                description = "Recording accelerometer data in the background"
                setShowBadge(true)
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
                enableVibration(false)
                enableLights(false)
            }
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
            
            // For Android 13+, ensure notification is visible
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                // Check if notifications are enabled
                if (!notificationManager.areNotificationsEnabled()) {
                    // Notifications are disabled, but we still try to show
                }
            }
        }
    }

    private fun createNotification(status: String = "Recording accelerometer data..."): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE
        )

        val elapsedTime = if (startTime > 0) {
            val seconds = (System.currentTimeMillis() - startTime) / 1000
            "${seconds}s"
        } else {
            "0s"
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Accelerometer Recorder")
            .setContentText("$status | Records: $recordCount | Time: $elapsedTime")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true) // Persistent notification that can't be dismissed
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)  // Changed from LOW to DEFAULT
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setShowWhen(true)
            .build()
    }

    private fun updateNotification(status: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, createNotification(status))
    }

    private fun getExternalStorageFile(fileName: String): File? {
        return try {
            // Check if we have permission to write to Downloads (Android 11+)
            val canWriteDownloads = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else {
                // Android 10 and below
                true // WRITE_EXTERNAL_STORAGE permission should be sufficient
            }
            
            if (canWriteDownloads) {
                // Try to use Downloads directory
                val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val chironDir = File(downloadsDir, "Chiron")
                if (!chironDir.exists()) {
                    val created = chironDir.mkdirs()
                    if (!created && !chironDir.exists()) {
                        throw java.io.IOException("Failed to create Downloads/Chiron directory")
                    }
                }
                
                // Test if we can actually write to this directory
                val testFile = File(chironDir, ".test_write")
                try {
                    testFile.writeText("test")
                    testFile.delete()
                } catch (e: Exception) {
                    // Can't write to Downloads, fall back to app storage
                    android.util.Log.w("AccelerometerService", "Cannot write to Downloads, using app storage: ${e.message}")
                    throw e
                }
                
                File(chironDir, fileName)
            } else {
                // No permission, use app's external files directory
                android.util.Log.w("AccelerometerService", "No Downloads permission, using app storage")
                val appDir = getExternalFilesDir(null) ?: filesDir
                val chironDir = File(appDir, "Chiron")
                if (!chironDir.exists()) {
                    chironDir.mkdirs()
                }
                File(chironDir, fileName)
            }
        } catch (e: java.io.IOException) {
            // Permission denied or other I/O error - use app storage
            android.util.Log.w("AccelerometerService", "Falling back to app storage due to: ${e.message}")
            val appDir = getExternalFilesDir(null) ?: filesDir
            val chironDir = File(appDir, "Chiron")
            if (!chironDir.exists()) {
                chironDir.mkdirs()
            }
            File(chironDir, fileName)
        } catch (e: Exception) {
            android.util.Log.e("AccelerometerService", "Error getting storage file: ${e.message}", e)
            // Final fallback to app's external files directory
            val appDir = getExternalFilesDir(null) ?: filesDir
            val chironDir = File(appDir, "Chiron")
            if (!chironDir.exists()) {
                chironDir.mkdirs()
            }
            File(chironDir, fileName)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Only stop recording if service is being destroyed
        // This handles cleanup when service is actually stopped
        if (isRecording) {
            sensorManager?.unregisterListener(sensorListener)
            serviceScope.launch {
                var dataToWrite: MinuteData? = null
                
                synchronized(minuteAggregationLock) {
                    // Prepare any remaining minute data for writing
                    currentMinuteData?.let { minuteData ->
                        if (minuteData.enmoValues.isNotEmpty()) {
                            // Copy data to write outside synchronized block
                            dataToWrite = MinuteData(
                                minuteData.minuteTimestamp,
                                ArrayList(minuteData.enmoValues)
                            )
                        }
                    }
                    currentMinuteData = null
                }
                
                // Write outside synchronized block
                dataToWrite?.let { writeMinuteData(it) }
                
                // Wait a bit for the write to complete
                kotlinx.coroutines.delay(100)
                
                try {
                    fileWriter?.flush()
                    fileWriter?.close()
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
        
        // Release wake lock if held
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
        wakeLock = null
    }

    fun getRecordCount(): Int = recordCount
    fun isRecording(): Boolean = isRecording
    fun getDataFile(): File? = dataFile

    companion object {
        const val CHANNEL_ID = "AccelerometerDataChannel"
        const val NOTIFICATION_ID = 1
        const val ACTION_START = "com.example.chiron.START_RECORDING"
        const val ACTION_STOP = "com.example.chiron.STOP_RECORDING"
    }
}

