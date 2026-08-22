"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// A plain divIcon avoids the classic Leaflet-in-webpack broken default
// marker icon problem (missing marker-icon.png paths) without needing to
// copy any image assets into the project.
const markerIcon = L.divIcon({
  className: "",
  html: `<div style="width:16px;height:16px;border-radius:50%;background:#38bdf8;border:3px solid #0a1b33;box-shadow:0 0 0 4px rgba(56,189,248,0.35);"></div>`,
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

function Recenter({ lat, lon }: { lat: number; lon: number }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo([lat, lon], map.getZoom(), { duration: 0.8 });
  }, [lat, lon, map]);
  return null;
}

export default function MonitoringMap({
  cityName,
  lat,
  lon,
}: {
  cityName: string;
  lat: number;
  lon: number;
}) {
  return (
    <div
      className="glass-card overflow-hidden"
      style={{
        height: 320,
        border: "1px solid rgba(56,189,248,0.28)",
        boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
      }}
    >
      <MapContainer
        center={[lat, lon]}
        zoom={11}
        scrollWheelZoom={false}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        <Marker position={[lat, lon]} icon={markerIcon}>
          <Popup>{cityName} monitoring location</Popup>
        </Marker>
        <Recenter lat={lat} lon={lon} />
      </MapContainer>
      {/* Leaflet injects tile <img> elements outside our JSX tree, so a
          scoped styled-jsx block can't reach them -- "global" is required
          here. Only lightens the map tiles; nothing else on the page. */}
      <style jsx global>{`
        .leaflet-tile-pane {
          filter: brightness(1.4) saturate(1.15) contrast(0.9);
        }
        .leaflet-container {
          background: #0d2038 !important;
        }
      `}</style>
    </div>
  );
}
