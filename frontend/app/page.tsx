import React from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import ChatInterface from "@/components/ChatInterface";
import ModelInfo from "@/components/ModelInfo";
import Disclaimer from "@/components/Disclaimer";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-1">
        <Hero />
        <ChatInterface />
        <ModelInfo />
        <Disclaimer />
      </main>
      <Footer />
    </div>
  );
}
