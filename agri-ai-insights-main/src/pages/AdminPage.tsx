import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, FileImage, TrendingUp, Target, Activity, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

const metrics = [
  { icon: Target, label: "Precision", value: "93.8%" },
  { icon: Activity, label: "Recall", value: "92.1%" },
  { icon: TrendingUp, label: "F1-Score", value: "92.9%" },
  { icon: CheckCircle2, label: "Accuracy", value: "94.2%" },
];

const recentPredictions = [
  { id: "P-1042", crop: "Tomato", disease: "Late Blight", confidence: "94.7%", status: "confirmed", date: "Apr 12, 2026" },
  { id: "P-1041", crop: "Potato", disease: "Early Blight", confidence: "89.3%", status: "confirmed", date: "Apr 12, 2026" },
  { id: "P-1040", crop: "Apple", disease: "Apple Scab", confidence: "91.2%", status: "pending", date: "Apr 11, 2026" },
  { id: "P-1039", crop: "Grape", disease: "Black Rot", confidence: "87.6%", status: "confirmed", date: "Apr 11, 2026" },
  { id: "P-1038", crop: "Corn", disease: "Healthy", confidence: "96.1%", status: "confirmed", date: "Apr 10, 2026" },
];

export default function AdminPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Admin Panel</h1>
        <p className="text-muted-foreground">Model performance metrics and experiment management</p>
      </div>

      {/* Metrics */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((m) => (
          <Card key={m.label} className="border-border/60">
            <CardContent className="pt-6 flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                <m.icon className="h-6 w-6 text-primary" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">{m.label}</p>
                <p className="text-2xl font-bold text-foreground">{m.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-3">
        <Button className="gap-2" onClick={() => toast.success("Report downloaded")}>
          <Download className="h-4 w-4" /> Download Report
        </Button>
        <Button variant="outline" className="gap-2" onClick={() => toast.success("Heatmap exported")}>
          <FileImage className="h-4 w-4" /> Export Heatmap
        </Button>
      </div>

      {/* Recent Predictions */}
      <Card className="border-border/60">
        <CardHeader>
          <CardTitle className="text-lg">Recent Predictions</CardTitle>
          <CardDescription>Latest diagnosis results across all clients</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Crop</TableHead>
                <TableHead>Disease</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recentPredictions.map((p) => (
                <TableRow key={p.id}>
                  <TableCell className="font-mono text-xs">{p.id}</TableCell>
                  <TableCell>{p.crop}</TableCell>
                  <TableCell className="font-medium">{p.disease}</TableCell>
                  <TableCell>{p.confidence}</TableCell>
                  <TableCell>
                    <Badge variant={p.status === "confirmed" ? "default" : "secondary"}>
                      {p.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm">{p.date}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Notes */}
      <Card className="border-border/60">
        <CardHeader>
          <CardTitle className="text-lg">Experiment Notes</CardTitle>
          <CardDescription>Record observations and results</CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Add notes about this experiment run, hyperparameter changes, observations..."
            className="min-h-[120px]"
          />
          <Button className="mt-3" size="sm" onClick={() => toast.success("Notes saved")}>
            Save Notes
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
