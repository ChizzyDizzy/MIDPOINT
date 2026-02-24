import React, { useState, useEffect } from "react";
import { getResources } from "../services/api";
import { Book, Phone, Globe, Users, ExternalLink } from "lucide-react";

const ResourceLink = ({ href, children }) => (
  <li>
    <a href={href} target="_blank" rel="noopener noreferrer" className="resource-link">
      {children}
      <ExternalLink size={12} />
    </a>
  </li>
);

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
    <div className="resource-panel">
      <h3>Helpful Resources</h3>
      <div style={{ marginBottom: "20px" }}>
        <h4 style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <Phone size={18} style={{ marginRight: "8px" }} />
          Emergency Contacts
        </h4>
        <ul className="resource-list">
          <ResourceLink href="https://988lifeline.org/">
            Suicide &amp; Crisis Lifeline: 988
          </ResourceLink>
          <ResourceLink href="https://www.crisistextline.org/">
            Crisis Text Line: Text HOME to 741741
          </ResourceLink>
          <ResourceLink href="https://www.samhsa.gov/find-help/national-helpline">
            SAMHSA Helpline: 1-800-662-4357
          </ResourceLink>
        </ul>
      </div>
      <div style={{ marginBottom: "20px" }}>
        <h4 style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <Book size={18} style={{ marginRight: "8px" }} />
          Self-Help Resources
        </h4>
        <ul className="resource-list">
          <ResourceLink href="https://www.calm.com/breathe">
            Breathing Exercises - Calm
          </ResourceLink>
          <ResourceLink href="https://www.headspace.com/meditation">
            Mindfulness Meditation - Headspace
          </ResourceLink>
          <ResourceLink href="https://www.therapistaid.com/therapy-worksheet/progressive-muscle-relaxation-script">
            Progressive Muscle Relaxation
          </ResourceLink>
          <ResourceLink href="https://www.psychologytoday.com/us/blog/prescriptions-life/201902/5-journaling-prompts-mental-health">
            Journaling Prompts
          </ResourceLink>
        </ul>
      </div>
      <div style={{ marginBottom: "20px" }}>
        <h4 style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <Users size={18} style={{ marginRight: "8px" }} />
          Support Groups
        </h4>
        <ul className="resource-list">
          <ResourceLink href="https://www.7cups.com/">
            7 Cups - Online Support Community
          </ResourceLink>
          <ResourceLink href="https://www.nami.org/Support-Education/Support-Groups">
            NAMI Support Groups
          </ResourceLink>
          <ResourceLink href="https://www.mentalhealthamerica.net/find-support-groups">
            Mental Health America - Find Groups
          </ResourceLink>
        </ul>
      </div>
      <div>
        <h4 style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <Globe size={18} style={{ marginRight: "8px" }} />
          Professional Help
        </h4>
        <ul className="resource-list">
          <ResourceLink href="https://www.psychologytoday.com/us/therapists">
            Find a Therapist - Psychology Today
          </ResourceLink>
          <ResourceLink href="https://www.betterhelp.com/">
            BetterHelp - Online Therapy
          </ResourceLink>
          <ResourceLink href="https://www.opencounseling.com/">
            Open Counseling - Free/Affordable
          </ResourceLink>
        </ul>
      </div>
    </div>
  );
};

export default ResourcePanel;
