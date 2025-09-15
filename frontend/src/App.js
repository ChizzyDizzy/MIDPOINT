import React, { useState } from "react";
import ChatInterface from "./components/ChatInterface";
import MoodTracker from "./components/MoodTracker";
import ResourcePanel from "./components/ResourcePanel";
import "./App.css";

function App() {
  const [showResources, setShowResources] = useState(false);
  const [moodData, setMoodData] = useState([]);

  return (
    <div className='App'>
      <header className='app-header'>
        <h1> SafeMind AI </h1>{" "}
        <p> Your Compassionate Mental Health Companion </p>{" "}
      </header>
      <div className='app-container'>
        <div className='main-content'>
          <ChatInterface
            onMoodUpdate={(mood) => setMoodData([...moodData, mood])}
            onEmergency={() => setShowResources(true)}
          />{" "}
        </div>
        <div className='sidebar'>
          <MoodTracker data={moodData} />{" "}
          <button
            className='resources-btn'
            onClick={() => setShowResources(!showResources)}
          >
            {showResources ? "Hide" : "Show"}
            Resources{" "}
          </button>{" "}
          {showResources && <ResourcePanel />}{" "}
        </div>{" "}
      </div>{" "}
    </div>
  );
}

export default App;
