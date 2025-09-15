import React, { useState, useEffect } from "react";
import { getResources } from "../services/api";
import { Book, Phone, Globe, Users } from "lucide-react";

const ResourcePanel = () => {
  const [resources, setResources] = useState(null);

  useEffect(() => {
    fetchResources();
  }, []);

  const fetchResources = async () => {
    try {
      const data = await getResources();
      setResources(data);
    } catch (error) {
      console.error("Error fetching resources:", error);
    }
  };

  return (
    <div className='resource-panel'>
      <h3> Helpful Resources </h3>
      <div style={{ marginBottom: "20px" }}>
        <h4
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: "10px",
          }}
        >
          <Phone size={18} style={{ marginRight: "8px" }} />
          Emergency Contacts{" "}
        </h4>{" "}
        <ul className='resource-list'>
          <li> Crisis Hotline: 1333 </li> <li> Mental Health: 1926 </li>{" "}
          <li> Emergency: 119 </li>{" "}
        </ul>{" "}
      </div>
      <div style={{ marginBottom: "20px" }}>
        <h4
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: "10px",
          }}
        >
          <Book size={18} style={{ marginRight: "8px" }} />
          Self - Help Resources{" "}
        </h4>{" "}
        <ul className='resource-list'>
          <li> Breathing Exercises </li> <li> Mindfulness Meditation </li>{" "}
          <li> Progressive Muscle Relaxation </li> <li> Journaling Prompts </li>{" "}
        </ul>{" "}
      </div>
      <div style={{ marginBottom: "20px" }}>
        <h4
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: "10px",
          }}
        >
          <Users size={18} style={{ marginRight: "8px" }} />
          Support Groups{" "}
        </h4>{" "}
        <ul className='resource-list'>
          <li> Online Support Communities </li> <li> Local Support Groups </li>{" "}
          <li> Family Counseling Services </li>{" "}
        </ul>{" "}
      </div>
      <div>
        <h4
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: "10px",
          }}
        >
          <Globe size={18} style={{ marginRight: "8px" }} />
          Professional Help{" "}
        </h4>{" "}
        <ul className='resource-list'>
          <li> Find a Therapist </li> <li> Psychiatrist Directory </li>{" "}
          <li> Counseling Centers </li>{" "}
        </ul>{" "}
      </div>{" "}
    </div>
  );
};

export default ResourcePanel;
