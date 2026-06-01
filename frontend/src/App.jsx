import { useState, useCallback } from "react";
import {
  Upload,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Flame,
  BarChart3,
  Radio,
  Repeat,
  Bot,
  FileText,
  Info,
  Shield,
  Download,
} from "lucide-react";

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeHeatmapTab, setActiveHeatmapTab] = useState("ela");
  const [expandedAccordions, setExpandedAccordions] = useState({});
  const [analysisOptions, setAnalysisOptions] = useState({
    run_ela: true,
    run_jpeg: true,
    run_noise: true,
    run_copy_move: true,
    run_ai: true,
    run_metadata: true,
  });

  const handleImageSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const runAnalysis = useCallback(async () => {
    if (!selectedImage) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedImage);
      Object.entries(analysisOptions).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });

      const response = await fetch("/api/analyze-upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Analysis failed (${response.status}): ${errText}`);
      }

      const result = await response.json();
      setResults(result);
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedImage, analysisOptions]);

  const toggleAccordion = useCallback((key) => {
    setExpandedAccordions((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // Color for AUTHENTICITY scores (higher = better = greener)
  const getAuthenticityColor = (score) => {
    if (score >= 80) return "#22c55e";
    if (score >= 60) return "#84cc16";
    if (score >= 40) return "#f59e0b";
    if (score >= 20) return "#f97316";
    return "#ef4444";
  };

  // Color for SUSPICION scores (higher = worse = redder)
  // This is the inverse of authenticity colors
  const getSuspicionColor = (score) => {
    if (score >= 80) return "#ef4444";
    if (score >= 60) return "#f97316";
    if (score >= 40) return "#f59e0b";
    if (score >= 20) return "#84cc16";
    return "#22c55e";
  };

  const getStatusIcon = (suspicionScore) => {
    if (suspicionScore <= 20)
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (suspicionScore <= 50)
      return <AlertTriangle className="w-5 h-5 text-amber-500" />;
    return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const branchLabels = {
    ela: "Adaptive ELA",
    jpeg_artifacts: "JPEG Artifacts",
    noise_analysis: "Noise Analysis",
    copy_move: "Copy-Move",
    ai_detection: "AI Detection",
    metadata: "Metadata",
  };

  const branchIconMap = {
    ela: Flame,
    jpeg_artifacts: BarChart3,
    noise_analysis: Radio,
    copy_move: Repeat,
    ai_detection: Bot,
    metadata: FileText,
  };

  const heatmapTabs = [
    { id: "ela", label: "ELA Heatmap", field: "ela.heatmap_base64" },
    {
      id: "jpeg_ghost",
      label: "JPEG Ghost",
      field: "jpeg.ghost_map_base64",
    },
    {
      id: "copy_move",
      label: "Copy-Move",
      field: "copy_move.heatmap_base64",
    },
    {
      id: "ai_detection",
      label: "AI Detection",
      field: "ai_detection.heatmap_base64",
    },
    {
      id: "frequency",
      label: "Frequency Spectrum",
      field: "ai_detection.frequency_spectrum_base64",
    },
    {
      id: "noise",
      label: "Noise Map",
      field: "noise.noise_map_base64",
    },
  ];

  const getNestedValue = (obj, path) => {
    const [parent, child] = path.split(".");
    return obj?.[parent]?.[child];
  };

  return (
    <div className="min-h-screen bg-gray-50 font-inter">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white sticky top-0 z-10">
        <div className="max-w-[1600px] mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900 tracking-tight">
                  Media Authenticity Verifier
                </h1>
                <p className="text-xs text-gray-500">
                  Multi-Branch Forensic Analysis for Digital Evidence
                </p>
              </div>
            </div>
            {results && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <span>ID: {results.analysis_id?.slice(0, 8)}...</span>
                <span className="text-gray-300">|</span>
                <span>{results.scoring?.analysis_timestamp?.replace("T", " ").slice(0, 19)}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-[1600px] mx-auto px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ====== LEFT PANEL: Upload & Options ====== */}
          <div className="lg:col-span-1 space-y-5">
            {/* Upload */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Upload className="w-4 h-4 text-blue-600" />
                Upload Image
              </h2>
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer bg-gray-50"
                onClick={() => document.getElementById("file-input").click()}
              >
                {imagePreview ? (
                  <div className="space-y-2">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-44 object-contain rounded-lg bg-white"
                    />
                    <p className="text-xs text-gray-400">
                      Click to change image
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Upload className="w-10 h-10 text-gray-300 mx-auto" />
                    <div>
                      <p className="text-gray-900 font-medium text-sm">
                        Drop image here
                      </p>
                      <p className="text-xs text-gray-500">
                        or click to browse
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
            </div>

            {/* Analysis Options */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">
                Analysis Options
              </h2>
              <div className="space-y-2.5">
                {Object.entries(analysisOptions).map(([key, value]) => {
                  const branchKey = key.replace("run_", "");
                  const Icon = branchIconMap[branchKey];
                  return (
                    <label
                      key={key}
                      className="flex items-center gap-3 cursor-pointer group"
                    >
                      <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) =>
                          setAnalysisOptions((prev) => ({
                            ...prev,
                            [key]: e.target.checked,
                          }))
                        }
                        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500"
                      />
                      {Icon && <Icon className="w-4 h-4 text-gray-400" />}
                      <span className="text-sm text-gray-700 group-hover:text-gray-900">
                        {branchLabels[branchKey] ||
                          branchKey.replace(/_/g, " ").toUpperCase()}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={runAnalysis}
              disabled={!selectedImage || loading}
              className="w-full bg-blue-600 text-white py-3.5 px-6 rounded-lg font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Analyzing...
                </span>
              ) : (
                "Run Forensic Analysis"
              )}
            </button>

            {/* Error */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                <div className="flex items-start gap-2">
                  <XCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-red-800">
                      Analysis Failed
                    </p>
                    <p className="text-xs text-red-600 mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Score Legend */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-gray-900 mb-3">
                Score Interpretation
              </h2>
              <div className="space-y-2">
                {[
                  { color: "#22c55e", label: "0-20: Authentic" },
                  { color: "#84cc16", label: "21-40: Likely Authentic" },
                  { color: "#f59e0b", label: "41-60: Suspicious" },
                  { color: "#f97316", label: "61-80: Likely Manipulated" },
                  { color: "#ef4444", label: "81-100: Manipulated" },
                ].map(({ color, label }) => (
                  <div key={label} className="flex items-center gap-3">
                    <div
                      className="w-3.5 h-3.5 rounded"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-xs text-gray-600">{label}</span>
                  </div>
                ))}
                <p className="text-xs text-gray-400 mt-2">
                  Suspicion scores: lower = cleaner
                </p>
              </div>
            </div>
          </div>

          {/* ====== RIGHT PANEL: Results ====== */}
          <div className="lg:col-span-2 space-y-5">
            {!results ? (
              <div className="bg-white border border-gray-200 rounded-xl p-16 text-center">
                <div className="w-16 h-16 bg-gray-50 border border-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Info className="w-8 h-8 text-gray-300" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  No Analysis Yet
                </h3>
                <p className="text-sm text-gray-500 max-w-sm mx-auto">
                  Upload an image and run the forensic analysis to see detailed
                  results across 6 forensic branches
                </p>
              </div>
            ) : (
              <>
                {/* Score Gauge */}
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <div className="flex flex-col items-center">
                    <div className="relative w-56 h-28 mb-4">
                      <svg
                        viewBox="0 0 200 100"
                        className="w-full h-full"
                      >
                        <path
                          d="M 10 90 A 90 90 0 0 1 190 90"
                          fill="none"
                          stroke="#E5E7EB"
                          strokeWidth="16"
                        />
                        <path
                          d="M 10 90 A 90 90 0 0 1 190 90"
                          fill="none"
                          stroke={getAuthenticityColor(
                            results.scoring.authenticity_score,
                          )}
                          strokeWidth="16"
                          strokeDasharray={`${results.scoring.authenticity_score * 2.83} 283`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center pt-6">
                        <div
                          className="text-5xl font-semibold tracking-tight"
                          style={{
                            color: getAuthenticityColor(
                              results.scoring.authenticity_score,
                            ),
                          }}
                        >
                          {results.scoring.authenticity_score}
                        </div>
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-semibold text-gray-900 tracking-tight">
                        {results.scoring.classification}
                      </div>
                      <div className="text-xs text-gray-500 mt-0.5">
                        Authenticity Score (0-100)
                      </div>
                    </div>
                  </div>
                </div>

                {/* Branch Score Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                  {Object.entries(results.scoring.branch_scores).map(
                    ([key, data]) => {
                      const Icon = branchIconMap[key];
                      const suspicionScore = data.score;
                      return (
                        <div
                          key={key}
                          className="bg-white border border-gray-200 rounded-xl p-4 hover:border-gray-300 transition-colors"
                        >
                          <div className="flex items-center justify-between mb-2.5">
                            <div className="flex items-center gap-2">
                              {Icon && (
                                <Icon className="w-4 h-4 text-blue-600" />
                              )}
                              <span className="font-semibold text-gray-900 text-sm">
                                {branchLabels[key]}
                              </span>
                            </div>
                            {getStatusIcon(suspicionScore)}
                          </div>
                          <div className="flex items-end gap-2 mb-2.5">
                            <div
                              className="text-2xl font-semibold tracking-tight"
                              style={{
                                color: getSuspicionColor(suspicionScore),
                              }}
                            >
                              {suspicionScore}
                            </div>
                            <div className="text-xs text-gray-400 pb-0.5 font-medium">
                              / 100 suspicion
                            </div>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-1.5">
                            <div
                              className="h-1.5 rounded-full transition-all"
                              style={{
                                width: `${suspicionScore}%`,
                                backgroundColor:
                                  getSuspicionColor(suspicionScore),
                              }}
                            />
                          </div>
                        </div>
                      );
                    },
                  )}
                </div>

                {/* Heatmap Tabs */}
                <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                  <div className="flex overflow-x-auto border-b border-gray-200">
                    {heatmapTabs.map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveHeatmapTab(tab.id)}
                        className={`px-5 py-2.5 font-medium text-xs whitespace-nowrap transition-colors border-b-2 -mb-[1px] ${
                          activeHeatmapTab === tab.id
                            ? "text-gray-900 border-blue-600"
                            : "text-gray-500 border-transparent hover:text-gray-700"
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <div className="p-4 bg-gray-50">
                    {heatmapTabs.map((tab) => {
                      if (activeHeatmapTab !== tab.id) return null;
                      const base64Data = getNestedValue(results, tab.field);
                      return (
                        <div
                          key={tab.id}
                          className="bg-white border border-gray-200 rounded-lg p-3"
                        >
                          {base64Data ? (
                            <img
                              src={`data:image/png;base64,${base64Data}`}
                              alt={tab.label}
                              className="w-full h-auto rounded"
                            />
                          ) : (
                            <div className="text-center py-12 text-sm text-gray-400">
                              No heatmap data available for this analysis
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Concerns & Recommendations */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="bg-white border border-gray-200 rounded-xl p-5">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-500" />
                      Primary Concerns
                    </h3>
                    <div className="space-y-2.5">
                      {results.scoring.primary_concerns?.length > 0 ? (
                        results.scoring.primary_concerns.map(
                          (concern, index) => {
                            const text =
                              typeof concern === "string"
                                ? concern
                                : concern.description || JSON.stringify(concern);
                            const severity = text
                              .toLowerCase()
                              .includes("critical")
                              ? "CRITICAL"
                              : text.toLowerCase().includes("high")
                                ? "HIGH"
                                : text.toLowerCase().includes("moderate")
                                  ? "MODERATE"
                                  : "INFO";
                            const severityColors = {
                              CRITICAL:
                                "bg-red-50 text-red-700 border border-red-200",
                              HIGH: "bg-orange-50 text-orange-700 border border-orange-200",
                              MODERATE:
                                "bg-amber-50 text-amber-700 border border-amber-200",
                              INFO: "bg-blue-50 text-blue-700 border border-blue-200",
                            };
                            return (
                              <div
                                key={index}
                                className="flex items-start gap-2"
                              >
                                <span
                                  className={`px-2 py-0.5 rounded-full text-xs font-semibold ${severityColors[severity]} shrink-0 mt-0.5`}
                                >
                                  {severity}
                                </span>
                                <p className="text-xs text-gray-600">{text}</p>
                              </div>
                            );
                          },
                        )
                      ) : (
                        <p className="text-xs text-gray-400">
                          No significant concerns detected
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="bg-white border border-gray-200 rounded-xl p-5">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      Recommendations
                    </h3>
                    <ol className="space-y-2.5">
                      {results.scoring.recommendations?.map((rec, index) => (
                        <li
                          key={index}
                          className="flex gap-2 text-xs text-gray-600"
                        >
                          <span className="text-blue-600 font-semibold shrink-0">
                            {index + 1}.
                          </span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>

                {/* Branch Detail Accordions */}
                <div className="space-y-2">
                  {/* ELA Detail */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => toggleAccordion("ela")}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <Flame className="w-4 h-4 text-blue-600" />
                        <span className="font-semibold text-gray-900 text-sm">
                          Adaptive ELA Details
                        </span>
                      </div>
                      {expandedAccordions.ela ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                    {expandedAccordions.ela && results.ela && (
                      <div className="px-5 py-3.5 border-t border-gray-200 grid grid-cols-3 gap-4 bg-gray-50">
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Quality Level
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.ela.selected_quality}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Mean Error
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.ela.primary_mean_error?.toFixed(4)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Anomalous Regions
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.ela.num_anomalous_regions}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* JPEG Detail */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => toggleAccordion("jpeg")}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <BarChart3 className="w-4 h-4 text-blue-600" />
                        <span className="font-semibold text-gray-900 text-sm">
                          JPEG Artifacts Details
                        </span>
                      </div>
                      {expandedAccordions.jpeg ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                    {expandedAccordions.jpeg && results.jpeg && (
                      <div className="px-5 py-3.5 border-t border-gray-200 grid grid-cols-2 gap-4 bg-gray-50">
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Is JPEG
                          </div>
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold border ${results.jpeg.is_jpeg ? "bg-green-50 text-green-700 border-green-200" : "bg-gray-50 text-gray-500 border-gray-200"}`}
                          >
                            {results.jpeg.is_jpeg ? "Yes" : "No"}
                          </span>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Double Compression
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.jpeg.double_compression > 0.01
                              ? `${(results.jpeg.double_compression * 100).toFixed(0)}% confidence`
                              : "Not detected"}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Quality Estimate
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.jpeg.compression_quality_estimate}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            QT Inconsistency
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.jpeg.quantization_inconsistency
                              ? "Detected"
                              : "None"}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Noise Detail */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => toggleAccordion("noise")}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <Radio className="w-4 h-4 text-blue-600" />
                        <span className="font-semibold text-gray-900 text-sm">
                          Noise Analysis Details
                        </span>
                      </div>
                      {expandedAccordions.noise ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                    {expandedAccordions.noise && results.noise && (
                      <div className="px-5 py-3.5 border-t border-gray-200 grid grid-cols-3 gap-4 bg-gray-50">
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Overall Score
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.noise.overall_noise_score?.toFixed(1)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Inconsistency
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.noise.noise_inconsistency_score?.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Frequency Anomaly
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.noise.frequency_anomaly_score?.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Copy-Move Detail */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => toggleAccordion("copy_move")}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <Repeat className="w-4 h-4 text-blue-600" />
                        <span className="font-semibold text-gray-900 text-sm">
                          Copy-Move Details
                        </span>
                      </div>
                      {expandedAccordions.copy_move ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                    {expandedAccordions.copy_move && results.copy_move && (
                      <div className="px-5 py-3.5 border-t border-gray-200 grid grid-cols-2 gap-4 bg-gray-50">
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Clone Detected
                          </div>
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold border ${results.copy_move.clone_detected ? "bg-red-50 text-red-700 border-red-200" : "bg-green-50 text-green-700 border-green-200"}`}
                          >
                            {results.copy_move.clone_detected ? "Yes" : "No"}
                          </span>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Confidence
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {(results.copy_move.clone_confidence * 100).toFixed(
                              0,
                            )}
                            %
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Regions Found
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.copy_move.num_clone_regions}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Method Used
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.copy_move.method_used?.join(", ") || "N/A"}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* AI Detection Detail */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => toggleAccordion("ai")}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <Bot className="w-4 h-4 text-blue-600" />
                        <span className="font-semibold text-gray-900 text-sm">
                          AI Detection Details
                        </span>
                      </div>
                      {expandedAccordions.ai ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                    {expandedAccordions.ai && results.ai_detection && (
                      <div className="px-5 py-3.5 border-t border-gray-200 space-y-3 bg-gray-50">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-xs font-medium text-gray-500 mb-0.5">
                              AI Likelihood
                            </div>
                            <div className="text-gray-900 font-medium text-sm">
                              {/* FIX: API already returns 0-100, don't multiply again */}
                              {results.ai_detection.ai_generated_likelihood?.toFixed(
                                1,
                              )}
                              %
                            </div>
                          </div>
                          <div>
                            <div className="text-xs font-medium text-gray-500 mb-0.5">
                              Possible Generator
                            </div>
                            <div className="text-gray-900 font-medium text-sm">
                              {results.ai_detection.possible_generator ||
                                "Unknown"}
                            </div>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="text-xs font-semibold text-gray-500">
                            Sub-Scores
                          </div>
                          {[
                            {
                              label: "EfficientNet",
                              value: results.ai_detection.efficientnet_score,
                              color: "bg-blue-600",
                            },
                            {
                              label: "CLIP Zero-Shot",
                              value: results.ai_detection.clip_score,
                              color: "bg-purple-600",
                            },
                            {
                              label: "Frequency",
                              value: results.ai_detection.frequency_score,
                              color: "bg-pink-600",
                            },
                            {
                              label: "Noise Pattern",
                              value: results.ai_detection.noise_score,
                              color: "bg-cyan-600",
                            },
                          ].map(({ label, value, color }) => (
                            <div key={label}>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-500 font-medium">
                                  {label}
                                </span>
                                <span className="text-gray-900 font-semibold">
                                  {value?.toFixed(1)}
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-1.5">
                                <div
                                  className={`${color} h-1.5 rounded-full transition-all`}
                                  style={{
                                    width: `${Math.min(100, value || 0)}%`,
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Metadata Detail */}
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => toggleAccordion("metadata")}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <FileText className="w-4 h-4 text-blue-600" />
                        <span className="font-semibold text-gray-900 text-sm">
                          Metadata Details
                        </span>
                      </div>
                      {expandedAccordions.metadata ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                    {expandedAccordions.metadata && results.metadata && (
                      <div className="px-5 py-3.5 border-t border-gray-200 grid grid-cols-2 gap-4 bg-gray-50">
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Camera
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.metadata.camera_make &&
                            results.metadata.camera_model
                              ? `${results.metadata.camera_make} ${results.metadata.camera_model}`
                              : "Unknown"}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            EXIF Present
                          </div>
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold border ${results.metadata.exif_present ? "bg-green-50 text-green-700 border-green-200" : "bg-red-50 text-red-700 border-red-200"}`}
                          >
                            {results.metadata.exif_present ? "Yes" : "No"}
                          </span>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Editing Software
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.metadata.editing_software || "None"}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Known Editor
                          </div>
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold border ${results.metadata.is_known_editor ? "bg-orange-50 text-orange-700 border-orange-200" : "bg-green-50 text-green-700 border-green-200"}`}
                          >
                            {results.metadata.is_known_editor ? "Yes" : "No"}
                          </span>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            GPS
                          </div>
                          <div className="text-gray-900 font-medium text-sm">
                            {results.metadata.gps_present
                              ? results.metadata.gps_coordinates
                              : "Not available"}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-0.5">
                            Timestamp Consistency
                          </div>
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold border ${results.metadata.timestamp_inconsistency ? "bg-red-50 text-red-700 border-red-200" : "bg-green-50 text-green-700 border-green-200"}`}
                          >
                            {results.metadata.timestamp_inconsistency
                              ? "Inconsistent"
                              : "Consistent"}
                          </span>
                        </div>
                        {results.metadata.stripping_detected && (
                          <div className="col-span-2">
                            <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-700">
                              Metadata stripping detected — EXIF data may have
                              been intentionally removed
                            </div>
                          </div>
                        )}
                        {results.metadata.warnings?.length > 0 && (
                          <div className="col-span-2">
                            <div className="text-xs font-medium text-gray-500 mb-1">
                              Warnings
                            </div>
                            <ul className="space-y-1">
                              {results.metadata.warnings.map((w, i) => (
                                <li
                                  key={i}
                                  className="text-xs text-gray-500 flex items-start gap-1"
                                >
                                  <span className="text-amber-400 shrink-0">
                                    •
                                  </span>
                                  {w}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Forensic Report Image */}
                {results.report_image_base64 && (
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                    <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                        <Download className="w-4 h-4 text-blue-600" />
                        Forensic Report
                      </h3>
                      <a
                        href={`data:image/png;base64,${results.report_image_base64}`}
                        download={`forensic_report_${results.analysis_id?.slice(0, 8)}.png`}
                        className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                      >
                        Download PNG
                      </a>
                    </div>
                    <div className="p-4 bg-gray-50">
                      <img
                        src={`data:image/png;base64,${results.report_image_base64}`}
                        alt="Forensic Report"
                        className="w-full h-auto rounded border border-gray-200"
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white mt-8">
        <div className="max-w-[1600px] mx-auto px-8 py-4 text-center text-xs text-gray-400">
          Media Authenticity Verifier v1.0 — Multi-Branch Forensic Analysis
          Pipeline — Developed for IFSO Delhi Digital Evidence Screening
        </div>
      </footer>
    </div>
  );
}
