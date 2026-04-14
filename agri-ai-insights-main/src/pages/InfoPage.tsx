import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database, Cpu, Lock, Eye } from "lucide-react";

const infoCards = [
  {
    icon: Database,
    title: "Dataset",
    badge: "PlantVillage",
    description: "38 disease classes across 14 crop species with over 54,000 expertly curated leaf images. Augmented with rotation, flipping, and color jittering.",
    details: [
      { label: "Images", value: "54,305" },
      { label: "Classes", value: "38" },
      { label: "Crops", value: "14" },
      { label: "Split", value: "80/10/10" },
    ],
  },
  {
    icon: Cpu,
    title: "Model Architecture",
    badge: "ResNet50",
    description: "Pre-trained ResNet-50 backbone with custom classification head. Fine-tuned using federated averaging across distributed client nodes.",
    details: [
      { label: "Parameters", value: "25.6M" },
      { label: "Input Size", value: "224×224" },
      { label: "Optimizer", value: "SGD" },
      { label: "LR", value: "0.001" },
    ],
  },
  {
    icon: Lock,
    title: "Privacy Method",
    badge: "Flower",
    description: "Federated Learning implemented via the Flower framework. Model updates are aggregated centrally — raw data never leaves the client device.",
    details: [
      { label: "Framework", value: "Flower 1.5" },
      { label: "Strategy", value: "FedAvg" },
      { label: "Rounds", value: "100" },
      { label: "Min Clients", value: "3" },
    ],
  },
  {
    icon: Eye,
    title: "Explainability",
    badge: "Grad-CAM",
    description: "Gradient-weighted Class Activation Mapping generates visual heatmaps highlighting discriminative regions that influenced the model's prediction.",
    details: [
      { label: "Method", value: "Grad-CAM" },
      { label: "Target Layer", value: "layer4" },
      { label: "Colormap", value: "Jet" },
      { label: "Resolution", value: "7×7 → 224" },
    ],
  },
];

export default function InfoPage() {
  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Dataset & Model Information</h1>
        <p className="text-muted-foreground">Technical specifications of the XFedCrop system</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {infoCards.map((card) => (
          <Card key={card.title} className="border-border/60 hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <card.icon className="h-5 w-5 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{card.title}</CardTitle>
                </div>
                <Badge variant="secondary">{card.badge}</Badge>
              </div>
              <CardDescription className="mt-2">{card.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                {card.details.map((d) => (
                  <div key={d.label} className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">{d.label}</p>
                    <p className="text-sm font-semibold text-foreground">{d.value}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
