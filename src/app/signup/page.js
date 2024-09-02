"use client";
import Navbar from "@/components/Navbar"
import { useState } from "react"
import {createUserWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/components/firebase";
import { useRouter } from "next/navigation";

export default function SignUp(){
    const router = useRouter();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault(); 
        setEmail("");
        setPassword("");
        await createUserWithEmailAndPassword(auth, email, password).then(() =>{ console.log("user created!!")
        router.push("/signin")
        }
        
    ).catch((e) => console.log(`${e.code} => ${e.message}`))
    }

    return (
        <div>
            <Navbar />
            <form className="pt-24" onSubmit={(e) => handleSubmit(e)}>
                <div className="w-full h-[85vh] flex flex-col gap-6 justify-center items-center">
                    <div className="grid grid-cols-2">
                        <div className="text-primary font-staat text-7xl flex justify-center items-center">
                            <h1>
                                REGISTER<br /> YOURSELF <br />TO CONTRIBUTE
                            </h1>
                        </div>
                        <div className="w-[25vw] flex flex-col gap-16 justify-center items-center p-7 text-primary  font-staat">
                            <div className="flex flex-row-reverse justify-between w-full items-center text-2xl">
                                <input type="email" id="email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-[15vw] text-primary p-2 rounded-lg font-mono font-bold border-2 border-primary"/>
                                <label htmlFor="email" className>email</label>
                            </div>
                            <div className="flex flex-row-reverse justify-between w-full items-center text-2xl">
                                <input type="password" id="password" value={password} minLength={7} onChange={(e) => setPassword(e.target.value)} className="w-[15vw] p-2 text-primary rounded-lg border-2 border-primary"/>
                                <label htmlFor="password">password</label>
                            </div>
                        </div>
                    </div>
                    
                    <div className="bg-primary text-white font-staat rounded-lg shadow-md hover:scale-125 hover:bg-white hover:text-primary transition-all">
                        <button className="px-3 py-2" type="submit">SignUp</button>
                    </div>
                </div>
            </form>
            
        </div>
    )
}