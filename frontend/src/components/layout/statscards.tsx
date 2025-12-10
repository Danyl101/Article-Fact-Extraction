import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FactStats } from "@/types/facts";

export default function StatCards({stats}:{stats:FactStats}){
    return(
        <div className="grid grid-cols-5 gap-4 my-4">
            <Card><CardContent> Articles: {stats.articles}</CardContent></Card>
            <Card><CardContent> Common Facts: {stats.common}</CardContent></Card>
            <Card><CardContent> Unique: {stats.unique}</CardContent></Card>
            <Card>
                <CardContent className="text-red-500">
                    Contradictions:{stats.contradictions}
                </CardContent>
            </Card>
            <Card><CardContent> Reliability:{stats.reliability}%</CardContent></Card>
        </div>
    )
}

