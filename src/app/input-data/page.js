"use client";
import Navbar from "@/components/Navbar";
import  Select  from "@mui/material/Select";
import  MenuItem  from "@mui/material/MenuItem";
import { auth, db } from "../../components/firebase";
import { useRouter } from "next/navigation";
import { serverTimestamp, addDoc, collection, Timestamp } from "firebase/firestore";
import { useState } from "react";
import { useSelector } from "react-redux";

export default function InputData(){
    const uid = useSelector(state => state["auth"]["Token"]);
    const router = useRouter();

    const [age, setAge] = useState("")
    const [gender, setGender] = useState("");
    const [totalCholestrol, setTotalCholestrol] = useState("");
    const [hdlCholestrol, setHdlCholestrol] = useState("");
    const [systollicBP, setSystollicBP] = useState("");
    const [diastollicBP, setDiastollicBP] = useState("");
    const [diabetes, setDiabetes] = useState("");
    const [smoking, setSmoking] = useState("");


    const handleAgeChange = (event) => setAge(event.target.value)
    const handleGenderChange = (event) => setGender(event.target.value)
    const handleTotalCholestrolChange = (event) => setTotalCholestrol(event.target.value)
    const handleHDLCholestrolChange = (event) => setHdlCholestrol(event.target.value)
    const handleSystollicBPChange = (event) => setSystollicBP(event.target.value)
    const handleDiastollicBPChange = (event) => setDiastollicBP(event.target.value)
    const handleDiabetesChange = (event) => setDiabetes(event.target.value)
    const handleSmokingChange = (event) => setSmoking(event.target.value)

    const handleFormSubmit = async (event) => {
        event.preventDefault();
        if (uid) {
            const response = await addDoc(collection(db, "riskFactor"), {
                uid: uid,
                age: age,
                gender: gender,
                total_cholestrol: totalCholestrol,
                hdl_cholestrol: hdlCholestrol,
                systollic_bp: systollicBP,
                diastollic_bp: diastollicBP,
                diabetes: diabetes,
                smoking: smoking,
                Timestamp: serverTimestamp(),
            });
            router.push("/thankYou")
        }
    }


    return (
        <div>
            <Navbar />
            <form className="pt-24 px-28 flex flex-col gap-28" onSubmit={(e) => handleFormSubmit(e)}>
                <div className="grid grid-cols-3 grid-rows-3 w-full gap-44" >
                    <div className="grid grid-cols-2 Age gap-36">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Age
                        </div>      
                        <Select value={age} onChange={handleAgeChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"30-34"}>30-34</MenuItem>
                            <MenuItem value={"35-39"}>35-39</MenuItem>
                            <MenuItem value={"40-44"}>40-44</MenuItem>
                            <MenuItem value={"45-49"}>45-49</MenuItem>
                            <MenuItem value={"50-54"}>50-54</MenuItem>
                            <MenuItem value={"55-59"}>55-59</MenuItem>
                            <MenuItem value={"60-64"}>60-64</MenuItem>
                            <MenuItem value={"65-69"}>65-69</MenuItem>
                            <MenuItem value={"70-74"}>40-44</MenuItem>  
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Gender
                        </div>      
                        <Select value={gender} onChange={handleGenderChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"Male"}>Male</MenuItem>
                            <MenuItem value={"Female"}>Female</MenuItem>
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Total Cholestrol
                        </div>      
                        <Select value={totalCholestrol} onChange={handleTotalCholestrolChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"< 4.1"}>lesser than 4.1</MenuItem>
                            <MenuItem value={"4.1 - 5.1"}>4.1 - 5.1</MenuItem>
                            <MenuItem value={"5.2 - 6.2"}>5.2 - 6.2</MenuItem>
                            <MenuItem value={"6.3 - 7.1"}>6.3 - 7.1</MenuItem>
                            <MenuItem value={"> 7.1"}>greater than 7.1</MenuItem>
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36 text-center">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            HDL <br/> Cholestrol
                        </div>      
                        <Select value={hdlCholestrol} onChange={handleHDLCholestrolChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"< 0.9"}>lesser than 0.9</MenuItem>
                            <MenuItem value={"0.9 - 1.16"}>0.9 - 1.16</MenuItem>
                            <MenuItem value={"1.17 - 1.29"}>1.17 - 1.29</MenuItem>
                            <MenuItem value={"1.30 - 1.55"}>1.30 - 1.55</MenuItem>
                            <MenuItem value={"> 1.56"}>greater than 1.56</MenuItem>
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36 text-center">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Systollic <br/> BP
                        </div>      
                        <Select value={systollicBP} onChange={handleSystollicBPChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"< 0.9"}>lesser than 0.9</MenuItem>
                            <MenuItem value={"0.9 - 1.16"}>0.9 - 1.16</MenuItem>
                            <MenuItem value={"1.17 - 1.29"}>1.17 - 1.29</MenuItem>
                            <MenuItem value={"1.30 - 1.55"}>1.30 - 1.55</MenuItem>
                            <MenuItem value={"> 1.56"}>greater than 1.56</MenuItem>
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36 text-center">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Diastollic <br/> BP
                        </div>      
                        <Select value={diastollicBP} onChange={handleDiastollicBPChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"< 80"}>lesser than 80</MenuItem>
                            <MenuItem value={"80 - 84"}>80 - 84</MenuItem>
                            <MenuItem value={"85 - 89"}>85 - 89</MenuItem>
                            <MenuItem value={"90 - 99"}>90 - 99</MenuItem>
                            <MenuItem value={"> 100"}>greater than 100</MenuItem>
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36 text-center">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Diabetes
                        </div>      
                        <Select value={diabetes} onChange={handleDiabetesChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"Yes"}>Yes</MenuItem>
                            <MenuItem value={"No"}>No</MenuItem>
                        </Select>
                    </div>
                    <div className="grid grid-cols-2 Gender gap-36 text-center">
                        <div className="text-xl text-primary font-staat flex items-center justify-center">
                            Smoking
                        </div>      
                        <Select value={smoking} onChange={handleSmokingChange} labelId="demo-simple-select-label" label="Age" className="w-36" required>
                            <MenuItem value={"Yes"}>Yes</MenuItem>
                            <MenuItem value={"No"}>No</MenuItem>
                        </Select>
                    </div>
                    
                </div>
                <div className="w-full flex justify-center">
                        <div className="">
                            <button className="rounded-lg px-3 py-2 bg-primary text-white font-staat hover:scale-125 hover:bg-white hover:text-primary transition-all">Submit</button>
                        </div>
                    </div>
            </form>
        </div>
    )
}