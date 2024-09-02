"use client";
import Image from "next/image";
import Navbar from "@/components/Navbar";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  return (
    <div className="">
      <Navbar />
      <div className="flex flex-col gap-28 pt-24">
        <div className="grid grid-cols-2 px-7">
          <div className="font-staat text-primary flex justify-center items-center text-8xl text-center">
            Protecting Your Health Through Data
          </div>
          <div className="flex justify-center items-center">
            <img src="./About.png" className="w-[75%] rounded-lg shadow-lg" />
          </div>
        </div>
        <div className="about p-10 flex flex-col gap-7 items-start">
            <div className="text-5xl font-semibold font-staat text-primary">
              About Us
            </div>
            <div className="text-2xl font-medium font-staat text-primary w-[50vw]">
            Welcome to HeartSense, a dedicated platform aimed at improving heart health through data-driven insights. Our mission is to collect and analyze critical information that can help identify risk factors for heart disease, enabling better prevention, early detection, and personalized care.
            </div>
        </div>
        <div className="about p-10 flex flex-col gap-7 items-end text-end">
            <div className="text-5xl font-semibold font-staat text-primary">
              Our Mission
            </div>
            <div className="text-2xl font-medium font-staat text-primary w-[50vw]">
            Heart disease remains one of the leading causes of death worldwide, but with the right knowledge and tools, many cases can be prevented. Our mission is to empower individuals, healthcare professionals, and researchers by providing a comprehensive database of risk factors related to heart disease. By gathering and analyzing this data, we aim to contribute to more accurate prediction models, ultimately reducing the burden of heart disease.
            </div>
        </div>
        <div className="about p-10 flex flex-col gap-7 items-start text-start">
            <div className="text-5xl font-semibold font-staat text-primary">
              What do we do?  
            </div>
            <div className="text-2xl font-medium font-staat text-primary w-[50vw]">
            At HeartSense, we focus on collecting data from a wide range of sources, including lifestyle habits, medical history, genetic information, and environmental factors. We provide a secure and user-friendly platform where users can contribute their data anonymously, knowing that they are playing a vital role in advancing heart disease research.
            </div>
        </div>
        <div className="about p-10 flex flex-col gap-7 items-end text-end">
            <div className="text-5xl font-semibold font-staat text-primary">
              Our Commitment
            </div>
            <div className="text-2xl font-medium font-staat text-primary w-[50vw]">
            We are committed to maintaining the highest standards of privacy and data security. All data collected on our platform is anonymized and encrypted, ensuring that your personal information is protected. Our team of experts follows strict ethical guidelines to ensure that the data is used solely for research and development purposes aimed at improving heart health.
            </div>
        </div>
        <div className="flex flex-col justify-center items-center py-24 px-16 text-center gap-10">
          <div className="grid grid-cols-2 gap-10 shadow-lg p-5">
              <div className=" text-8xl font-staat text-primary font-bold flex justify-center items-center border-r-4 border-primary">
                <h1>Join Us in the Fight Against Heart Disease</h1>
              </div>
              <div className="text-3xl font-staat font-mediun text-primary flex items-center">
              By participating in our data collection efforts, you are contributing to a global initiative to combat heart disease. Together, we can create a future where heart disease is detected early and prevented before it becomes life-threatening. Whether you are a patient, a healthcare provider, or simply someone interested in heart health, your contribution matters.
          </div>
          </div>
          <div>
            <button className="underline text-primary text-3xl font-staat" onClick={() => router.push("/signup")}>Sign Up Now</button>
          </div>
        </div>
      </div>
    </div>    
  );
}
