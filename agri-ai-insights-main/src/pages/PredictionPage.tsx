import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Upload, ImageIcon, Loader2, CheckCircle2, AlertTriangle, Leaf } from "lucide-react";
import { toast } from "sonner";

type PredictionState = "empty" | "uploaded" | "loading" | "result" | "error";

interface PredictionResult {
  label: string;
  class_index: number;
  confidence: number;
  probabilities_top5: Array<{
    class_index: number;
    label: string;
    confidence: number;
  }>;
}

const API_BASE = "http://127.0.0.1:8000";

export default function PredictionPage() {
  const [state, setState] = useState<PredictionState>("empty");
  const [preview, setPreview] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      toast.error("Please upload an image file");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
      setFile(f);
      setState("uploaded");
    };
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const runDiagnosis = async () => {
    if (!file) {
      toast.error("No file selected");
      return;
    }

    setState("loading");
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const data: PredictionResult = await res.json();
      setResult(data);
      setState("result");
      // Store prediction data globally for other pages
      (window as any).__lastPredictionFile = file;
      (window as any).__lastPredictionResult = data;
      localStorage.setItem("__lastPredictionFile", file.name);
      localStorage.setItem("__lastPredictionResult", JSON.stringify(data));
      toast.success("Diagnosis complete!");
    } catch (err) {
      toast.error((err as Error).message || "Error running diagnosis");
      setState("error");
    }
  };

  const reset = () => {
    setState("empty");
    setPreview(null);
    setFile(null);
    setResult(null);
  };

  const diseaseToSeverity = (disease: string): string => {
    const diseaseUpper = disease.toUpperCase();
    if (diseaseUpper.includes("HEALTHY")) return "Healthy";
    if (diseaseUpper.includes("BLIGHT") || diseaseUpper.includes("RUST")) return "High";
    if (diseaseUpper.includes("SPOT") || diseaseUpper.includes("MILDEW")) return "Medium";
    return "Low";
  };

  const diseaseToTreatment = (disease: string): string => {
    const diseaseUpper = disease.toUpperCase();
    if (diseaseUpper.includes("HEALTHY")) return "No treatment needed. Maintain proper leaf hygiene and spacing.";
    if (diseaseUpper.includes("BLIGHT"))
      return "Apply copper-based fungicide immediately. Remove and destroy infected leaves. Ensure proper spacing for air circulation.";
    if (diseaseUpper.includes("RUST"))
      return "Apply sulfur-based fungicide. Remove heavily infected leaves. Improve air circulation around plants.";
    if (diseaseUpper.includes("MILDEW"))
      return "Apply powdery mildew fungicide. Increase air flow. Avoid overhead watering. Remove infected plant parts.";
    if (diseaseUpper.includes("SPOT"))
      return "Apply copper or chlorothalonil fungicide. Remove infected leaves. Avoid wetting foliage. Prune for airflow.";
    return "Consult with a local agricultural extension office for specific treatment recommendations.";
  };

  const diseaseToTrait = (disease: string): string => {
    const diseaseUpper = disease.toUpperCase();
    if (diseaseUpper.includes("APPLE")) return "Apple";
    if (diseaseUpper.includes("TOMATO")) return "Tomato";
    if (diseaseUpper.includes("POTATO")) return "Potato";
    if (diseaseUpper.includes("CORN") || diseaseUpper.includes("MAIZE")) return "Corn";
    if (diseaseUpper.includes("GRAPE")) return "Grape";
    if (diseaseUpper.includes("PEPPER")) return "Pepper";
    if (diseaseUpper.includes("BLUEBERRY")) return "Blueberry";
    return "Unknown Crop";
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Disease Prediction</h1>
        <p className="text-muted-foreground">Upload a leaf image to diagnose crop diseases</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Upload Card */}
        <Card className="border-border/60">
          <CardHeader>
            <CardTitle className="text-lg">Leaf Image</CardTitle>
            <CardDescription>Drag and drop or click to upload</CardDescription>
          </CardHeader>
          <CardContent>
            {state === "empty" ? (
              <label
                className={`flex flex-col items-center justify-center h-64 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
                  dragActive
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50 hover:bg-muted/30"
                }`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
              >
                <Upload className="h-10 w-10 text-muted-foreground mb-3" />
                <p className="text-sm text-muted-foreground mb-1">Drop your leaf image here</p>
                <p className="text-xs text-muted-foreground">PNG, JPG up to 10MB</p>
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                />
              </label>
            ) : (
              <div className="relative">
                {preview && (
                  <div className="h-64 rounded-xl overflow-hidden bg-muted flex items-center justify-center">
                    <img src={preview} alt="Uploaded leaf" className="h-full w-full object-contain" />
                  </div>
                )}
                {!preview && (
                  <div className="h-64 rounded-xl bg-muted flex items-center justify-center">
                    <ImageIcon className="h-16 w-16 text-muted-foreground/30" />
                  </div>
                )}
                <div className="mt-4 flex gap-3">
                  {state === "uploaded" && (
                    <Button onClick={runDiagnosis} className="flex-1 gap-2">
                      <Leaf className="h-4 w-4" /> Run Diagnosis
                    </Button>
                  )}
                  {state === "loading" && (
                    <Button disabled className="flex-1 gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" /> Analyzing...
                    </Button>
                  )}
                  {(state === "result" || state === "error") && (
                    <Button onClick={runDiagnosis} className="flex-1 gap-2">
                      <Leaf className="h-4 w-4" /> Re-analyze
                    </Button>
                  )}
                  <Button variant="outline" onClick={reset}>
                    Reset
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Result Card */}
        <Card className="border-border/60">
          <CardHeader>
            <CardTitle className="text-lg">Diagnosis Result</CardTitle>
            <CardDescription>AI-powered disease analysis</CardDescription>
          </CardHeader>
          <CardContent>
            {state === "empty" || state === "uploaded" ? (
              <div className="h-64 flex flex-col items-center justify-center text-muted-foreground">
                <ImageIcon className="h-12 w-12 mb-3 opacity-30" />
                <p className="text-sm">Upload an image and run diagnosis</p>
              </div>
            ) : state === "loading" ? (
              <div className="h-64 flex flex-col items-center justify-center gap-4">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                <div className="text-center">
                  <p className="text-sm font-medium text-foreground">Running federated model...</p>
                  <p className="text-xs text-muted-foreground mt-1">Analyzing disease patterns</p>
                </div>
                <Progress value={65} className="w-48 h-2" />
              </div>
            ) : state === "error" ? (
              <div className="h-64 flex flex-col items-center justify-center text-red-500 gap-3">
                <AlertTriangle className="h-10 w-10" />
                <p className="text-sm font-medium">Error running diagnosis</p>
                <Button onClick={reset} variant="outline" size="sm">
                  Try Again
                </Button>
              </div>
            ) : result ? (
              <div className="space-y-5 animate-fade-in-up">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-primary" />
                    <span className="font-semibold text-foreground">{result.label}</span>
                  </div>
                  <Badge variant="destructive" className="gap-1">
                    <AlertTriangle className="h-3 w-3" /> {diseaseToSeverity(result.label)}
                  </Badge>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1.5">
                    <span className="text-muted-foreground">Confidence</span>
                    <span className="font-medium text-foreground">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <Progress value={result.confidence * 100} className="h-2.5" />
                </div>
                <div className="text-sm">
                  <span className="text-muted-foreground">Crop: </span>
                  <span className="font-medium text-foreground">{diseaseToTrait(result.label)}</span>
                </div>
                <div className="p-4 rounded-xl bg-primary/5 border border-primary/10">
                  <p className="text-sm font-medium text-foreground mb-1">Treatment Recommendation</p>
                  <p className="text-sm text-muted-foreground">{diseaseToTreatment(result.label)}</p>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
