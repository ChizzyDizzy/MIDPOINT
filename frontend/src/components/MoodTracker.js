import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const MoodTracker = ({ data = [] }) => {
  // Convert mood data for chart
  const chartData = data.slice(-10).map((item, index) => ({
    time: index + 1,
    mood:
      item.sentiment === "positive"
        ? 5
        : item.sentiment === "neutral"
        ? 3
        : item.sentiment === "concerned"
        ? 2
        : 1,
  }));

  return (
    <div className='mood-tracker'>
      <h3> Mood Tracking </h3>{" "}
      {chartData.length > 0 ? (
        <ResponsiveContainer width='100%' height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray='3 3' />
            <XAxis dataKey='time' />
            <YAxis domain={[0, 5]} /> <Tooltip />
            <Line
              type='monotone'
              dataKey='mood'
              stroke='#667eea'
              strokeWidth={2}
            />{" "}
          </LineChart>{" "}
        </ResponsiveContainer>
      ) : (
        <p style={{ color: "#999", textAlign: "center", padding: "20px" }}>
          Mood tracking will appear as you chat{" "}
        </p>
      )}{" "}
    </div>
  );
};

export default MoodTracker;
