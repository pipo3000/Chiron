package com.example.chiron

import android.content.Context
import android.os.Environment
import org.json.JSONObject
import java.io.File

data class CosinorAgeResult(
    val success: Boolean,
    val message: String,
    val cosinorAge: Double? = null,
    val sourceFile: File? = null
)

/**
 * Utility which locates the latest daily accelerometer CSV (accelerometer_YYYYMMDD.csv)
 * stored by the background service in Downloads/Chiron and uses Chaquopy to run
 * CosinorAge prediction on-device using the cosinorage Python package.
 * 
 * Package reference: https://github.com/ADAMMA-CDHI-ETH-Zurich/CosinorAge
 * 
 * Note: Uses reflection to access Chaquopy classes to avoid IDE import issues.
 * The Chaquopy runtime is included by the plugin at build time.
 */
object CosinorAgePredictor {
    
    @Volatile
    private var pythonInitialized = false
    private val initLock = Any()
    
    /**
     * Initialize Python runtime using reflection to access Chaquopy classes.
     * This avoids IDE import recognition issues while still working at runtime.
     */
    private fun initializePython(context: Context): Boolean {
        synchronized(initLock) {
            if (pythonInitialized) return true
            
            return try {
                // Use reflection to access Chaquopy classes
                val pythonClass = Class.forName("com.chaquo.python.Python")
                val androidPlatformClass = Class.forName("com.chaquo.python.android.AndroidPlatform")
                
                // Check if Python is already started
                val isStartedMethod = pythonClass.getMethod("isStarted")
                val isStarted = isStartedMethod.invoke(null) as Boolean
                
                if (!isStarted) {
                    // Create AndroidPlatform instance first
                    val platformConstructor = androidPlatformClass.getConstructor(Context::class.java)
                    val platform = platformConstructor.newInstance(context.applicationContext)
                    
                    // Python.start takes a Platform parameter - try using Object.class since
                    // Java reflection will accept any Object subclass at runtime
                    try {
                        // Try the standard approach: start(Object)
                        val startMethod = pythonClass.getMethod("start", Object::class.java)
                        startMethod.invoke(null, platform)
                    } catch (e: NoSuchMethodException) {
                        // If that fails, try to find any start method with one parameter
                        val startMethods = pythonClass.methods.filter { 
                            it.name == "start" && 
                            java.lang.reflect.Modifier.isStatic(it.modifiers) &&
                            it.parameterTypes.size == 1
                        }
                        
                        if (startMethods.isEmpty()) {
                            throw NoSuchMethodException("Python.start method not found")
                        }
                        
                        // Try each method until one works
                        var succeeded = false
                        var lastException: Exception? = null
                        for (method in startMethods) {
                            try {
                                method.invoke(null, platform)
                                succeeded = true
                                break
                            } catch (e: IllegalArgumentException) {
                                // Wrong parameter type, try next
                                lastException = e
                                continue
                            } catch (e: Exception) {
                                // Other error, might be the right method
                                lastException = e
                                continue
                            }
                        }
                        
                        if (!succeeded && lastException != null) {
                            throw lastException
                        }
                    }
                }
                
                pythonInitialized = true
                true
            } catch (e: ClassNotFoundException) {
                e.printStackTrace()
                pythonInitialized = false
                false
            } catch (e: Exception) {
                e.printStackTrace()
                pythonInitialized = false
                false
            }
        }
    }

    /**
     * Predicts CosinorAge using the provided file path.
     * Useful for tests where the file may be stored in app-internal storage.
     */
    fun predictCosinorAgeFromFile(
        context: Context,
        file: File,
        age: Int? = null,
        gender: String? = null
    ): CosinorAgeResult {
        if (!file.exists() || file.length() < 100) {
            return CosinorAgeResult(
                success = false,
                message = "File missing or insufficient data",
                sourceFile = file
            )
        }

        if (!initializePython(context)) {
            return CosinorAgeResult(
                success = false,
                message = "Failed to initialize Python runtime. Please check Chaquopy configuration.",
                sourceFile = file
            )
        }

        return try {
            val pythonClass = Class.forName("com.chaquo.python.Python")
            val getInstanceMethod = pythonClass.getMethod("getInstance")
            val python = getInstanceMethod.invoke(null)

            val getModuleMethod = python.javaClass.getMethod("getModule", String::class.java)
               // First try to test if cosinorage is available
               // This helps diagnose if the package is bundled
               try {
                   val testModuleMethod = python.javaClass.getMethod("getModule", String::class.java)
                   try {
                       val cosinorageModule = testModuleMethod.invoke(python, "cosinorage")
                       android.util.Log.d("CosinorAgePredictor", "cosinorage module found and loaded successfully")
                       
                       // Try to get module info for debugging
                       try {
                           val getAttrMethod = cosinorageModule.javaClass.getMethod("get", String::class.java)
                           val versionObj = getAttrMethod.invoke(cosinorageModule, "__version__")
                           android.util.Log.d("CosinorAgePredictor", "cosinorage version: ${versionObj?.toString() ?: "unknown"}")
                       } catch (e: Exception) {
                           android.util.Log.d("CosinorAgePredictor", "Could not get cosinorage version: ${e.message}")
                       }
                   } catch (e: java.lang.reflect.InvocationTargetException) {
                       // Unwrap Python exception
                       val cause = e.cause ?: e
                       val errorMsg = cause.message ?: e.message ?: "Unknown error"
                       android.util.Log.e("CosinorAgePredictor", "cosinorage module NOT available: $errorMsg", cause)
                       
                       // Check if it's an import error or module not found
                       if (errorMsg.contains("No module named") || errorMsg.contains("ModuleNotFoundError")) {
                           return CosinorAgeResult(
                               success = false,
                               message = "cosinorage package not bundled with app. The package must be installed in build.gradle python.pip block. Error: $errorMsg",
                               sourceFile = file
                           )
                       } else {
                           return CosinorAgeResult(
                               success = false,
                               message = "cosinorage package import failed: $errorMsg. Check if all dependencies (numpy, pandas, scipy) are available.",
                               sourceFile = file
                           )
                       }
                   } catch (e: Exception) {
                       android.util.Log.e("CosinorAgePredictor", "Error testing cosinorage module: ${e.message}", e)
                       // Continue - let the Python script handle the error
                   }
               } catch (e: Exception) {
                   android.util.Log.w("CosinorAgePredictor", "Could not test cosinorage module: ${e.message}")
               }
               
               val module = try {
                   android.util.Log.d("CosinorAgePredictor", "Attempting to load Python module 'cosinor_predictor'")
                   getModuleMethod.invoke(python, "cosinor_predictor")
               } catch (e: java.lang.reflect.InvocationTargetException) {
                // Unwrap the actual Python exception - this contains the real error
                val cause = e.cause ?: e
                val errorMessage = when {
                    cause is java.lang.Exception && cause.message != null -> cause.message!!
                    cause.message != null -> cause.message!!
                    else -> e.message ?: "Unknown error"
                }
                
                // Log full stack trace for debugging
                android.util.Log.e("CosinorAgePredictor", "Failed to load Python module 'cosinor_predictor'", cause)
                android.util.Log.e("CosinorAgePredictor", "Error details: $errorMessage")
                
                // Check for common error patterns
                val userMessage = when {
                    errorMessage.contains("No module named") -> {
                        "Python module not found. This might indicate:\n" +
                        "1. The Python file is missing\n" +
                        "2. A Python import error (e.g., missing cosinorage package)\n" +
                        "Error: $errorMessage"
                    }
                    errorMessage.contains("ImportError") || errorMessage.contains("ModuleNotFoundError") -> {
                        "Python import error. Make sure all dependencies are installed.\n" +
                        "Error: $errorMessage"
                    }
                    else -> {
                        "Python module failed to load.\nError: $errorMessage"
                    }
                }
                
                return CosinorAgeResult(
                    success = false,
                    message = userMessage,
                    sourceFile = file
                )
            } catch (e: Exception) {
                android.util.Log.e("CosinorAgePredictor", "Error loading Python module: ${e.message}", e)
                return CosinorAgeResult(
                    success = false,
                    message = "Error loading Python module: ${e.message}",
                    sourceFile = file
                )
            }

            val methods = module.javaClass.methods
            val callAttrMethod = methods.find {
                it.name == "callAttr" &&
                    it.parameterTypes.size >= 1 &&
                    it.parameterTypes[0] == String::class.java &&
                    (it.parameterTypes.size == 1 || it.isVarArgs)
            } ?: throw NoSuchMethodException("callAttr method not found")

            // For varargs methods called via reflection, wrap arguments in array
            val resultObj = if (callAttrMethod.isVarArgs) {
                // Varargs method - wrap all arguments after method name in Object array
                callAttrMethod.invoke(
                    module,
                    "predict_from_file",
                    arrayOf<Any>(
                        file.absolutePath,
                        age ?: 40,
                        gender ?: "male"
                    )
                )
            } else {
                // Non-varargs - pass directly
                callAttrMethod.invoke(
                    module,
                    "predict_from_file",
                    file.absolutePath,
                    age ?: 40,
                    gender ?: "male"
                )
            }
            val resultJson = resultObj.toString()

            val json = JSONObject(resultJson)
            val success = json.getBoolean("success")
            val message = json.getString("message")
            val cosinorAge = if (json.has("cosinor_age") && !json.isNull("cosinor_age")) json.getDouble("cosinor_age") else null

            CosinorAgeResult(
                success = success,
                message = message,
                cosinorAge = cosinorAge,
                sourceFile = file
            )
        } catch (e: Exception) {
            println(e)
            CosinorAgeResult(
                success = false,
                message = "Prediction error: ${e.message}",
                sourceFile = file
            )
        }
    }

    private fun getChironDir(): File? {
        // Try Downloads first
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val chironDir = File(downloadsDir, "Chiron")
        if (chironDir.exists() && chironDir.isDirectory) {
            return chironDir
        }
        return null
    }
    
    private fun getChironDirFromAppStorage(context: Context): File? {
        // Fallback to app's external files directory
        val appDir = context.getExternalFilesDir(null) ?: context.filesDir
        val chironDir = File(appDir, "Chiron")
        return if (chironDir.exists() && chironDir.isDirectory) chironDir else null
    }
    
    private fun getAllChironDirs(context: Context): List<File> {
        val dirs = mutableListOf<File>()
        getChironDir()?.let { dirs.add(it) }
        getChironDirFromAppStorage(context)?.let { dirs.add(it) }
        return dirs
    }

    /**
     * Generates synthetic accelerometer data files for the last week if no files exist.
     * This is called automatically on app installation/first launch.
     * 
     * @param context Android context
     * @return Number of files generated
     */
    fun generateSyntheticDataIfNeeded(context: Context): Int {
        // Check if files already exist
        val existingFiles = mutableListOf<File>()
        getAllChironDirs(context).forEach { dir ->
            dir.listFiles { _, name ->
                name.startsWith("accelerometer_") && name.endsWith(".csv")
            }?.let { existingFiles.addAll(it.asList()) }
        }
        
        if (existingFiles.isNotEmpty()) {
            android.util.Log.d("CosinorAgePredictor", "Synthetic data generation skipped: ${existingFiles.size} files already exist")
            return 0
        }
        
        android.util.Log.d("CosinorAgePredictor", "No accelerometer files found. Generating synthetic data for the last week...")
        
        val calendar = java.util.Calendar.getInstance()
        val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", java.util.Locale.getDefault())
        val fileDateFormat = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.getDefault())
        
        // Get Chiron directory (try Downloads first, fallback to app storage)
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val chironDir = File(downloadsDir, "Chiron")
        
        // Try to create directory, fallback to app storage if permissions denied
        val targetDir = if (chironDir.exists() || chironDir.mkdirs()) {
            chironDir
        } else {
            val appDir = context.getExternalFilesDir(null) ?: context.filesDir
            File(appDir, "Chiron").apply { mkdirs() }
        }
        
        var generatedCount = 0
        
        // Generate files for the last 7 days (excluding today)
        for (daysAgo in 6 downTo 1) {
            calendar.time = java.util.Date()
            calendar.add(java.util.Calendar.DAY_OF_YEAR, -daysAgo)
            
            val targetDate = calendar.time
            val dateStr = fileDateFormat.format(targetDate)
            val fileName = "accelerometer_$dateStr.csv"
            val file = File(targetDir, fileName)
            
            try {
                file.bufferedWriter().use { writer ->
                    writer.write("timestamp,enmo\n")
                    
                    // Generate 1440 minutes (1 full day)
                    calendar.set(java.util.Calendar.HOUR_OF_DAY, 0)
                    calendar.set(java.util.Calendar.MINUTE, 0)
                    calendar.set(java.util.Calendar.SECOND, 0)
                    calendar.set(java.util.Calendar.MILLISECOND, 0)
                    
                    for (minute in 0 until 1440) {
                        val timestamp = dateFormat.format(calendar.time)
                        val hour = calendar.get(java.util.Calendar.HOUR_OF_DAY)
                        
                        // Generate realistic ENMO based on time of day
                        val enmo = generateRealisticEnmo(hour, calendar.get(java.util.Calendar.MINUTE))
                        writer.write("$timestamp,$enmo\n")
                        
                        calendar.add(java.util.Calendar.MINUTE, 1)
                    }
                }
                
                android.util.Log.d("CosinorAgePredictor", "Generated: $fileName")
                generatedCount++
            } catch (e: Exception) {
                android.util.Log.e("CosinorAgePredictor", "Failed to generate $fileName: ${e.message}", e)
            }
        }
        
        android.util.Log.d("CosinorAgePredictor", "Generated $generatedCount synthetic data files")
        return generatedCount
    }
    
    /**
     * Generates a realistic ENMO value based on time of day (circadian rhythm).
     */
    private fun generateRealisticEnmo(hour: Int, minute: Int): Double {
        val random = java.util.Random()
        val baseEnmo = when {
            hour >= 22 || hour < 6 -> 0.005 + random.nextDouble() * 0.015  // Sleep hours
            hour in 6..8 -> 0.03 + random.nextDouble() * 0.08  // Morning wake-up
            hour in 9..11 -> 0.05 + random.nextDouble() * 0.12  // Morning activity
            hour in 12..13 -> 0.04 + random.nextDouble() * 0.10  // Midday
            hour in 14..16 -> 0.06 + random.nextDouble() * 0.15  // Afternoon
            hour in 17..20 -> 0.08 + random.nextDouble() * 0.18  // Evening activity
            else -> 0.02 + random.nextDouble() * 0.06  // Winding down (21:00-22:00)
        }
        
        val variation = (random.nextDouble() - 0.5) * 0.02
        val enmo = (baseEnmo + variation).coerceAtLeast(0.0)
        return String.format(java.util.Locale.US, "%.6f", enmo).toDouble()
    }

    /**
     * Copies a test file from assets to the Downloads/Chiron directory.
     * Useful for testing with synthetic data.
     * 
     * @param context Android context
     * @param assetFileName Name of the file in assets (e.g., "accelerometer_20251030.csv")
     * @return The copied File, or null if failed
     */
    fun copyTestFileFromAssets(context: Context, assetFileName: String): File? {
        return try {
            // First check if asset exists
            val assetList = try {
                context.assets.list("")
            } catch (e: Exception) {
                android.util.Log.e("CosinorAgePredictor", "Failed to list assets: ${e.message}")
                null
            }
            
            if (assetList == null || !assetList.contains(assetFileName)) {
                android.util.Log.e("CosinorAgePredictor", "Asset file '$assetFileName' not found. Available assets: ${assetList?.joinToString()}")
                return null
            }
            
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val chironDir = File(downloadsDir, "Chiron")
            if (!chironDir.exists()) {
                val created = chironDir.mkdirs()
                if (!created) {
                    android.util.Log.e("CosinorAgePredictor", "Failed to create directory: ${chironDir.absolutePath}")
                }
            }
            
            val targetFile = File(chironDir, assetFileName)
            
            // Copy from assets
            context.assets.open(assetFileName).use { inputStream ->
                targetFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            
            android.util.Log.d("CosinorAgePredictor", "Successfully copied test file to: ${targetFile.absolutePath}")
            targetFile
        } catch (e: java.io.FileNotFoundException) {
            android.util.Log.e("CosinorAgePredictor", "Asset file not found: $assetFileName", e)
            null
        } catch (e: Exception) {
            android.util.Log.e("CosinorAgePredictor", "Error copying asset file: ${e.message}", e)
            e.printStackTrace()
            null
        }
    }

    /**
     * Collects all previous day files (excluding today) from all Chiron directories.
     */
    fun getAllPreviousDayFiles(context: Context, maxDays: Int? = 7): List<File> {
        // Only use Downloads/Chiron folder, not app storage
        val chironDir = getChironDir()
        if (chironDir == null || !chironDir.exists() || !chironDir.isDirectory) {
            android.util.Log.w("CosinorAgePredictor", "getAllPreviousDayFiles: Downloads/Chiron directory not found")
            return emptyList()
        }
        val dirs = listOf(chironDir)
        
        // Get today's date and calculate cutoff date (maxDays ago)
        val calendar = java.util.Calendar.getInstance()
        val today = calendar.time

        val cutoffDate = if (maxDays != null) {
            calendar.add(java.util.Calendar.DAY_OF_YEAR, -maxDays)
            calendar.time
        } else {
            null
        }
        
        val dateFormat = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.getDefault())
        val todayStr = dateFormat.format(today)
        val cutoffDateStr = cutoffDate?.let { dateFormat.format(it) } ?: "ALL"
        val todayFileName = "accelerometer_$todayStr.csv"
        
        android.util.Log.d("CosinorAgePredictor", "Filtering files: today=$todayStr, cutoff=$cutoffDateStr (maxDays=${maxDays ?: "ALL"})")
        
        // Collect files from all directories, excluding today's file and files older than maxDays
        val allPreviousFiles = mutableListOf<File>()
        for (dir in dirs) {
            if (!dir.exists() || !dir.isDirectory) {
                continue
            }
            
            val files = dir.listFiles { _, name ->
                name.startsWith("accelerometer_") && name.endsWith(".csv")
            } ?: continue
            
            for (file in files) {
                // Skip today's file
                if (file.name == todayFileName) {
                    continue
                }
                
                // Extract date from filename (format: accelerometer_YYYYMMDD.csv)
                try {
                    val dateStr = file.name.removePrefix("accelerometer_").removeSuffix(".csv")
                    if (dateStr.length == 8) {
                        val fileDate = dateFormat.parse(dateStr)
                        if (fileDate != null &&
                            (cutoffDate == null || fileDate >= cutoffDate) &&
                            fileDate < today) {
                            allPreviousFiles.add(file)
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.w("CosinorAgePredictor", "Could not parse date from filename: ${file.name}: ${e.message}")
                    // If date parsing fails, skip this file
                }
            }
        }
        
        // Deduplicate files by name (in case same file exists in multiple directories)
        val uniqueFiles = allPreviousFiles
            .groupBy { it.name }
            .map { (_, files) -> files.first() }  // Take the first occurrence of each filename
        
        // Sort by filename (which contains date) to ensure chronological order
        val sortedFiles = uniqueFiles.sortedBy { it.name }
        
        android.util.Log.d("CosinorAgePredictor", "Found ${allPreviousFiles.size} files, ${sortedFiles.size} unique: ${sortedFiles.map { it.name }}")
        
        return sortedFiles
    }

    /**
     * Returns all available accelerometer files (including today's partial file) for visualization.
     */
    fun getAllAvailableFiles(context: Context): List<File> {
        val chironDir = getChironDir()
        if (chironDir == null || !chironDir.exists() || !chironDir.isDirectory) {
            android.util.Log.w("CosinorAgePredictor", "getAllAvailableFiles: Downloads/Chiron directory not found")
            return emptyList()
        }

        val dirs = listOf(chironDir)
        val collected = mutableListOf<File>()

        for (dir in dirs) {
            if (!dir.exists() || !dir.isDirectory) continue

            val files = dir.listFiles { _, name ->
                name.startsWith("accelerometer_") && name.endsWith(".csv")
            } ?: continue

            collected.addAll(files)
        }

        val uniqueFiles = collected
            .groupBy { it.name }
            .map { (_, files) -> files.first() }

        return uniqueFiles.sortedBy { it.name }
    }
    
    /**
     * Concatenates multiple CSV files into a single file.
     * Assumes all files have the same format: timestamp,enmo
     * Skips header rows except for the first file.
     */
    fun concatenateCsvFiles(files: List<File>, outputFile: File): Boolean {
        if (files.isEmpty()) {
            android.util.Log.w("CosinorAgePredictor", "No files to concatenate")
            return false
        }
        
        return try {
            outputFile.outputStream().bufferedWriter().use { writer ->
                var isFirstFile = true
                
                for (file in files) {
                    if (!file.exists() || !file.canRead()) {
                        android.util.Log.w("CosinorAgePredictor", "Skipping unreadable file: ${file.name}")
                        continue
                    }
                    
                    file.inputStream().bufferedReader().use { reader ->
                        var isFirstLine = true
                        reader.forEachLine { line ->
                            val trimmedLine = line.trim()
                            
                            // Skip header row except for the first file
                            if (isFirstLine && trimmedLine.lowercase().startsWith("timestamp")) {
                                if (isFirstFile) {
                                    writer.write(line)
                                    writer.newLine()
                                }
                                isFirstLine = false
                            } else {
                                // Write data lines (skip empty lines)
                                if (trimmedLine.isNotEmpty() && trimmedLine.contains(",")) {
                                    // Only write if it looks like valid CSV data (contains comma)
                                    writer.write(line)
                                    writer.newLine()
                                }
                                isFirstLine = false
                            }
                        }
                    }
                    
                    isFirstFile = false
                }
            }
            
            android.util.Log.d("CosinorAgePredictor", "Successfully concatenated ${files.size} files into ${outputFile.name} (${outputFile.length()} bytes)")
            true
        } catch (e: Exception) {
            android.util.Log.e("CosinorAgePredictor", "Error concatenating CSV files: ${e.message}", e)
            false
        }
    }
    
    /**
     * Predicts CosinorAge using all previous day files concatenated together.
     * Useful for predictions where we want complete data from previous days.
     */
    fun predictCosinorAgeFromPreviousDays(
        context: Context,
        age: Int? = null,
        gender: String? = null,
        maxDays: Int? = 14
    ): CosinorAgeResult {
        android.util.Log.d("CosinorAgePredictor", "predictCosinorAgeFromPreviousDays: Starting search for previous day files (maxDays=${maxDays ?: "ALL"})")
        
        val files = getAllPreviousDayFiles(context, maxDays = maxDays)
        if (files.isEmpty()) {
            // Try to get all directories and list files for better error message
            val dirs = getAllChironDirs(context)
            val allFiles = mutableListOf<String>()
            dirs.forEach { dir ->
                dir.listFiles()?.forEach { file ->
                    if (file.name.startsWith("accelerometer_") && file.name.endsWith(".csv")) {
                        allFiles.add("${dir.name}/${file.name}")
                    }
                }
            }
            
            android.util.Log.w("CosinorAgePredictor", "No previous day files found. Available files: ${allFiles.joinToString()}")
            
            return CosinorAgeResult(
                success = false,
                message = if (allFiles.isNotEmpty()) {
                    "Found ${allFiles.size} file(s) but none from previous days (excluding today). Files: ${allFiles.take(3).joinToString()}"
                } else {
                    "No previous day's accelerometer file found (excluding today). Please ensure files exist in Downloads/Chiron/ or app storage."
                },
                sourceFile = null
            )
        }

        android.util.Log.d("CosinorAgePredictor", "Found ${files.size} previous day files: ${files.map { it.name }}")
        
        // Check if files have sufficient data
        val totalSize = files.sumOf { it.length() }
        if (totalSize < 100) {
            return CosinorAgeResult(
                success = false,
                message = "Insufficient data in previous day files. Please record more accelerometer data.",
                sourceFile = null
            )
        }

        // Create a temporary concatenated file
        val tempDir = context.cacheDir
        val concatenatedFile = File(tempDir, "concatenated_accelerometer_${System.currentTimeMillis()}.csv")
        
        if (!concatenateCsvFiles(files, concatenatedFile)) {
            return CosinorAgeResult(
                success = false,
                message = "Failed to concatenate previous day files for prediction.",
                sourceFile = null
            )
        }

        android.util.Log.d("CosinorAgePredictor", "Using concatenated file for prediction: ${concatenatedFile.name} (${concatenatedFile.length()} bytes, from ${files.size} files)")
        
        // Use the existing predictCosinorAgeFromFile method
        val result = predictCosinorAgeFromFile(context, concatenatedFile, age, gender)
        
        // Clean up temporary file
        try {
            if (concatenatedFile.exists()) {
                concatenatedFile.delete()
                android.util.Log.d("CosinorAgePredictor", "Cleaned up temporary concatenated file")
            }
        } catch (e: Exception) {
            android.util.Log.w("CosinorAgePredictor", "Could not delete temporary file: ${e.message}")
        }
        
        return result
    }
    
    /**
     * Gets visualization data (ENMO and cosinor fit) for plotting.
     */
    fun getVisualizationData(
        context: Context,
        file: File,
        age: Int? = null,
        gender: String? = null
    ): VisualizationDataResult {
        if (!file.exists() || file.length() < 100) {
            return VisualizationDataResult(
                success = false,
                message = "File missing or insufficient data"
            )
        }

        if (!initializePython(context)) {
            return VisualizationDataResult(
                success = false,
                message = "Failed to initialize Python runtime"
            )
        }

        return try {
            val pythonClass = Class.forName("com.chaquo.python.Python")
            val getInstanceMethod = pythonClass.getMethod("getInstance")
            val python = getInstanceMethod.invoke(null)

            val getModuleMethod = python.javaClass.getMethod("getModule", String::class.java)
            val module = getModuleMethod.invoke(python, "cosinor_predictor")

            val methods = module.javaClass.methods
            val callAttrMethod = methods.find {
                it.name == "callAttr" &&
                    it.parameterTypes.size >= 1 &&
                    it.parameterTypes[0] == String::class.java
            } ?: throw NoSuchMethodException("callAttr method not found")

            val resultObj = if (callAttrMethod.isVarArgs) {
                callAttrMethod.invoke(
                    module,
                    "get_visualization_data",
                    arrayOf<Any>(
                        file.absolutePath,
                        age ?: 40,
                        gender ?: "male"
                    )
                )
            } else {
                callAttrMethod.invoke(
                    module,
                    "get_visualization_data",
                    file.absolutePath,
                    age ?: 40,
                    gender ?: "male"
                )
            }
            val resultJson = resultObj.toString()

            val json = JSONObject(resultJson)
            val success = json.getBoolean("success")
            val message = json.getString("message")
            
            val timestamps = mutableListOf<String>()
            val enmoValues = mutableListOf<Double>()
            val cosinorFit = mutableListOf<Double>()
            
            if (json.has("timestamps")) {
                val timestampsArray = json.getJSONArray("timestamps")
                for (i in 0 until timestampsArray.length()) {
                    timestamps.add(timestampsArray.getString(i))
                }
            }
            
            if (json.has("enmo_values")) {
                val enmoArray = json.getJSONArray("enmo_values")
                for (i in 0 until enmoArray.length()) {
                    enmoValues.add(enmoArray.getDouble(i))
                }
            }
            
            if (json.has("cosinor_fit")) {
                val fitArray = json.getJSONArray("cosinor_fit")
                for (i in 0 until fitArray.length()) {
                    cosinorFit.add(fitArray.getDouble(i))
                }
            }
            
            val cosinorParams = mutableMapOf<String, Double>()
            if (json.has("cosinor_params")) {
                val paramsObj = json.getJSONObject("cosinor_params")
                if (paramsObj.has("mesor")) cosinorParams["mesor"] = paramsObj.getDouble("mesor")
                if (paramsObj.has("amplitude")) cosinorParams["amplitude"] = paramsObj.getDouble("amplitude")
                if (paramsObj.has("acrophase")) cosinorParams["acrophase"] = paramsObj.getDouble("acrophase")
            }

            val fitMetadata = mutableMapOf<String, String>()
            if (json.has("fit_metadata")) {
                val metadataObj = json.getJSONObject("fit_metadata")
                metadataObj.keys().forEach { key ->
                    fitMetadata[key] = metadataObj.getString(key)
                }
            }

            val rhythmRobustness = if (json.has("rhythm_robustness") && !json.isNull("rhythm_robustness")) {
                json.getDouble("rhythm_robustness")
            } else {
                null
            }

            VisualizationDataResult(
                success = success,
                message = message,
                timestamps = timestamps,
                enmoValues = enmoValues,
                cosinorFit = cosinorFit,
                cosinorParams = cosinorParams,
                fitMetadata = fitMetadata,
                rhythmRobustness = rhythmRobustness
            )
        } catch (e: Exception) {
            android.util.Log.e("CosinorAgePredictor", "Error getting visualization data: ${e.message}", e)
            VisualizationDataResult(
                success = false,
                message = "Error: ${e.message}"
            )
        }
    }
}

data class VisualizationDataResult(
    val success: Boolean,
    val message: String,
    val timestamps: List<String> = emptyList(),
    val enmoValues: List<Double> = emptyList(),
    val cosinorFit: List<Double> = emptyList(),
    val cosinorParams: Map<String, Double> = emptyMap(),
    val fitMetadata: Map<String, String> = emptyMap(),
    val rhythmRobustness: Double? = null
)
