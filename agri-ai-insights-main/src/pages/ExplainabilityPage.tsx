import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Eye, Info, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const API_BASE = "http://127.0.0.1:8000";

interface XAISummary {
  label: string;
  confidence: number;
  hotspot_ratio: number;
  center_focus_ratio: number;
  hotspot_region: string;
  attention_style?: string;
  confidence_band?: string;
  top_attention_regions?: string[];
  top_alternatives?: Array<{
    label: string;
    confidence: number;
    class_index: number;
  }>;
  explanation: string;
}

export default function ExplainabilityPage() {
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [xaiSummary, setXaiSummary] = useState<XAISummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasFile, setHasFile] = useState(false);
  const loadedFileKeyRef = useRef<string | null>(null);

  const getLastPredictionFile = (): File | undefined => {
    const w = window as unknown as { __lastPredictionFile?: File };
    return w.__lastPredictionFile;
  };

  useEffect(() => {
    const checkForPredictionFile = async () => {
      const file = getLastPredictionFile();
      if (!file) return;

      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (loadedFileKeyRef.current === key || loading) return;

      loadedFileKeyRef.current = key;
      if (!hasFile) {
        setHasFile(true);
      }
      await loadExplanation(file);
    };

    checkForPredictionFile();
    const interval = setInterval(checkForPredictionFile, 1000);
    return () => clearInterval(interval);
  }, [hasFile, loading]);

  const loadExplanation = async (file: File) => {
    setLoading(true);
    try {
      const gradcamForm = new FormData();
      gradcamForm.append("file", file);

      const summaryForm = new FormData();
      summaryForm.append("file", file);

      const [gradRes, xaiRes] = await Promise.all([
        fetch(`${API_BASE}/explain/gradcam`, {
          method: "POST",
          body: gradcamForm,
        }),
        fetch(`${API_BASE}/explain/summary`, {
          method: "POST",
          body: summaryForm,
        }),
      ]);

      if (gradRes.ok) {
        const blob = await gradRes.blob();
        setGradcamImage(URL.createObjectURL(blob));
      }

      if (xaiRes.ok) {
        const data: XAISummary = await xaiRes.json();
        setXaiSummary(data);
      }
    } catch (err) {
      console.error("Error loading explanation:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = () => {
    setGradcamImage(null);
    setXaiSummary(null);
    const file = getLastPredictionFile();
    if (file) {
      loadExplanation(file);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Explainability</h1>
        <p className="text-muted-foreground">Understand why the model made its prediction</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Heatmap Viewer */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="border-border/60">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <CardTitle className="text-lg">Grad-CAM Visualization</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-muted">
                {loading ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-muted/80">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : gradcamImage ? (
                  <img
                    src={gradcamImage}
                    alt="Grad-CAM"
                    className="w-full h-full object-contain"
                  />
                ) : !hasFile ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-3">
                    <p className="text-sm">No prediction yet.</p>
                    <p className="text-xs">Go to Prediction tab to upload an image.</p>
                  </div>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-red-500 gap-3">
                    <p className="text-sm font-medium">Failed to load Grad-CAM</p>
                    <Button size="sm" onClick={handleRetry}>Retry</Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/60">
            <CardContent className="pt-5 flex items-start gap-3">
              <Info className="h-5 w-5 text-accent mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-medium text-foreground mb-1">Clinical-style Explanation</p>
                <p className="text-sm text-muted-foreground">
                  {xaiSummary?.explanation ||
                    "Upload an image and run diagnosis to see the explanation."}
                </p>
                {xaiSummary?.top_attention_regions?.length ? (
                  <div className="mt-3 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">Top attended regions:</span>{" "}
                    {xaiSummary.top_attention_regions.join(" → ")}
                  </div>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Summary */}
        <div className="space-y-4">
          <Card className="border-border/60">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Eye className="h-5 w-5 text-primary" /> Attention Summary
              </CardTitle>
              <CardDescription>Model focus distribution</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {xaiSummary ? (
                <>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Hotspot Coverage</span>
                      <span className="font-medium text-foreground">
                        {(xaiSummary.hotspot_ratio * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={xaiSummary.hotspot_ratio * 100} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Center Focus</span>
                      <span className="font-medium text-foreground">
                        {xaiSummary.center_focus_ratio.toFixed(2)}x
                      </span>
                    </div>
                    <Progress
                      value={Math.min(xaiSummary.center_focus_ratio * 33.33, 100)}
                      className="h-2"
                    />
                  </div>
                  <div>
                    <div className="text-sm">
                      <span className="text-muted-foreground">Focused Region: </span>
                      <span className="font-medium text-foreground capitalize">
                        {xaiSummary.hotspot_region}
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm">
                      <span className="text-muted-foreground">Attention Style: </span>
                      <span className="font-medium text-foreground capitalize">
                        {(xaiSummary.attention_style || "n/a").replace("_", " ")}
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">No data available</p>
              )}
            </CardContent>
          </Card>

          <Card className="border-border/60">
            <CardHeader>
              <CardTitle className="text-lg">Prediction Confidence</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {xaiSummary ? (
                <>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">{xaiSummary.label}</span>
                    <span className="font-semibold text-foreground">
                      {(xaiSummary.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <Progress value={xaiSummary.confidence * 100} className="h-2.5" />
                  {xaiSummary.confidence_band ? (
                    <p className="text-xs text-muted-foreground">
                      Confidence band: <span className="capitalize">{xaiSummary.confidence_band.replace("_", " ")}</span>
                    </p>
                  ) : null}
                  {xaiSummary.top_alternatives?.length ? (
                    <div className="pt-2 border-t border-border/60">
                      <p className="text-xs font-medium text-foreground mb-1">Closest alternatives</p>
                      {xaiSummary.top_alternatives.slice(0, 2).map((alt) => (
                        <div key={alt.class_index} className="flex justify-between text-xs text-muted-foreground">
                          <span className="truncate pr-2">{alt.label}</span>
                          <span>{(alt.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </>
              ) : (
                <p className="text-xs text-muted-foreground">No data available</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
