import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Users, Radio, BarChart3, Activity, Wifi, WifiOff } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const stats = [
  { icon: Users, label: "Active Clients", value: "8", sub: "of 12 registered" },
  { icon: Radio, label: "Communication Round", value: "47 / 100", sub: "47% complete" },
  { icon: BarChart3, label: "Global Accuracy", value: "94.2%", sub: "+1.3% from last round" },
  { icon: Activity, label: "Training Progress", value: "47%", progress: 47 },
];

const accuracyData = Array.from({ length: 20 }, (_, i) => ({
  round: (i + 1) * 5,
  accuracy: Math.min(55 + i * 2.3 + Math.random() * 2 - 1, 95.5).toFixed(1),
}));

const clients = [
  { name: "Node A — East Africa", status: "online", samples: "12,400", lastSync: "2 min ago" },
  { name: "Node B — South Asia", status: "online", samples: "9,800", lastSync: "5 min ago" },
  { name: "Node C — Latin America", status: "online", samples: "8,200", lastSync: "12 min ago" },
  { name: "Node D — Southeast Asia", status: "offline", samples: "6,100", lastSync: "3 hrs ago" },
  { name: "Node E — North Africa", status: "online", samples: "11,300", lastSync: "1 min ago" },
  { name: "Node F — Europe (Research)", status: "online", samples: "15,600", lastSync: "8 min ago" },
];

export default function FederatedPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Federated Learning Insights</h1>
        <p className="text-muted-foreground">Monitor collaborative training across distributed nodes</p>
      </div>

      {/* Status Cards */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <Card key={s.label} className="border-border/60">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <s.icon className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">{s.label}</p>
                  <p className="text-xl font-bold text-foreground">{s.value}</p>
                </div>
              </div>
              {s.progress !== undefined ? (
                <Progress value={s.progress} className="h-2" />
              ) : (
                <p className="text-xs text-muted-foreground">{s.sub}</p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Chart */}
      <Card className="border-border/60">
        <CardHeader>
          <CardTitle className="text-lg">Accuracy Over Communication Rounds</CardTitle>
          <CardDescription>Global model performance convergence</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="round" tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" label={{ value: "Round", position: "insideBottom", offset: -5 }} />
                <YAxis domain={[50, 100]} tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" label={{ value: "Accuracy %", angle: -90, position: "insideLeft" }} />
                <Tooltip contentStyle={{ borderRadius: "0.75rem", border: "1px solid hsl(var(--border))", background: "hsl(var(--card))" }} />
                <Line type="monotone" dataKey="accuracy" stroke="hsl(var(--primary))" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Client Cards */}
      <div>
        <h2 className="text-lg font-semibold text-foreground mb-4">Client Participation</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {clients.map((c) => (
            <Card key={c.name} className="border-border/60 hover:shadow-md transition-shadow">
              <CardContent className="pt-5">
                <div className="flex items-center justify-between mb-3">
                  <p className="font-medium text-foreground text-sm">{c.name}</p>
                  <Badge variant={c.status === "online" ? "default" : "secondary"} className="gap-1 text-xs">
                    {c.status === "online" ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
                    {c.status}
                  </Badge>
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{c.samples} samples</span>
                  <span>Last sync: {c.lastSync}</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
