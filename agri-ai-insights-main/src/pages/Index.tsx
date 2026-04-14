import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Upload,
  Search,
  BarChart3,
  Shield,
  Lock,
  Brain,
  Zap,
  Leaf,
  ArrowRight,
  Github,
  BookOpen,
} from "lucide-react";

const trustItems = [
  { icon: Shield, title: "Federated Learning", desc: "Collaborative training across distributed nodes" },
  { icon: Lock, title: "Privacy Preservation", desc: "Data never leaves the local device" },
  { icon: Brain, title: "Explainable AI", desc: "Grad-CAM visual explanations for every prediction" },
  { icon: Zap, title: "Real-time Diagnosis", desc: "Instant disease detection from leaf images" },
];

const steps = [
  { icon: Upload, title: "Upload Image", desc: "Drag and drop or select a leaf image from your device" },
  { icon: Search, title: "Predict Disease", desc: "Our federated model analyzes the image for disease patterns" },
  { icon: BarChart3, title: "View Explanation", desc: "See Grad-CAM heatmaps showing what influenced the prediction" },
];

export default function Index() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navbar */}
      <nav className="border-b bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Leaf className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold text-foreground">XFedCrop</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-foreground transition-colors">How It Works</a>
            <Link to="/dashboard/prediction">
              <Button size="sm">Open Dashboard</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="py-24 md:py-32 px-6">
        <div className="max-w-4xl mx-auto text-center animate-fade-in-up">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-8">
            <Shield className="h-4 w-4" />
            Privacy-Preserving Agricultural AI
          </div>
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-foreground leading-tight mb-6">
            Explainable Federated{" "}
            <span className="text-primary">Crop Disease</span>{" "}
            Diagnosis
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
            A collaborative deep learning system that diagnoses crop diseases while keeping your data private.
            Powered by federated learning and explainable AI.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/dashboard/prediction">
              <Button size="lg" className="gap-2 text-base px-8">
                <Upload className="h-5 w-5" /> Upload Leaf Image
              </Button>
            </Link>
            <a href="#how-it-works">
              <Button size="lg" variant="outline" className="gap-2 text-base px-8">
                View System Demo <ArrowRight className="h-4 w-4" />
              </Button>
            </a>
          </div>
        </div>
      </section>

      {/* Trust Section */}
      <section id="features" className="py-20 px-6 bg-secondary/40">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-2xl md:text-3xl font-bold text-center mb-4 text-foreground">Built for Trust & Transparency</h2>
          <p className="text-muted-foreground text-center max-w-xl mx-auto mb-14">
            Every component is designed with privacy, accuracy, and scientific rigor in mind.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {trustItems.map((item) => (
              <Card key={item.title} className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1 border-border/60">
                <CardContent className="pt-8 pb-6 px-6 text-center">
                  <div className="mx-auto w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mb-5 group-hover:bg-primary/20 transition-colors">
                    <item.icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="font-semibold text-foreground mb-2">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl md:text-3xl font-bold text-center mb-4 text-foreground">How It Works</h2>
          <p className="text-muted-foreground text-center max-w-lg mx-auto mb-14">
            Three simple steps from image upload to actionable insights.
          </p>
          <div className="grid md:grid-cols-3 gap-8">
            {steps.map((step, i) => (
              <div key={step.title} className="relative text-center animate-fade-in" style={{ animationDelay: `${i * 150}ms` }}>
                <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-5">
                  <step.icon className="h-8 w-8 text-primary" />
                </div>
                <div className="absolute -top-2 -left-2 w-8 h-8 rounded-full bg-primary text-primary-foreground text-sm font-bold flex items-center justify-center md:left-1/2 md:-translate-x-[3.5rem]">
                  {i + 1}
                </div>
                <h3 className="font-semibold text-foreground mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-12 px-6 bg-card">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <Leaf className="h-5 w-5 text-primary" />
            <span className="font-semibold text-foreground">XFedCrop</span>
            <span className="text-sm text-muted-foreground ml-2">© 2026</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <a href="#" className="hover:text-foreground transition-colors flex items-center gap-1">
              <BookOpen className="h-4 w-4" /> Documentation
            </a>
            <a href="#" className="hover:text-foreground transition-colors flex items-center gap-1">
              <Github className="h-4 w-4" /> GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
