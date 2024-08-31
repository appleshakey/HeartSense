"use client";
import { useState } from "react";
import Navbar from "@/components/Navbar"
import {signInWithEmailAndPassword} from "firebase/auth";
import { auth } from "@/components/firebase";
import { useDispatch } from "react-redux";
import { Login } from "@/state/auth/authSlice";
import { useRouter } from "next/navigation";

export default function SignIn(){
    const router = useRouter();
    const dispatch = useDispatch();
    const [email, setEmail] = useState(""); 
    const [password, setPassword] = useState("");

    const handleSubmit = (e) => {
        e.preventDefault();
        setEmail("");
        setPassword("");
        signInWithEmailAndPassword(auth, email, password).then((UserCredential) => {
            dispatch(Login(UserCredential["user"]["uid"]))
            router.push("/input-data")
        })
        
    }

    return (
        <div>
            <Navbar />
            
            <div className="pt-28">
                <form className="w-full h-[85vh] flex flex-col gap-10 justify-center items-center" onSubmit={(e) => handleSubmit(e)}>
                    <div className="flex gap-10 bg-primary p-10 rounded-xl shadow-lg">
                        <div className="text-7xl text-white font-staat  ">
                            Log In To <br />Monitor <br />Health
                        </div>
                        <div className="flex flex-col justify-center items-center text-white gap-10 font-staat w-[20vw]">
                            <div className="flex flex-row-reverse justify-between w-full items-center">
                                <input type="email" id="email" value={email} onChange={(e) => setEmail(e.target.value)} className="text-primary font-mono font-bold p-2 rounded-lg"/>
                                <label className="text-2xl">email</label>
                            </div>
                            <div className="flex flex-row-reverse justify-between w-full items-center">
                                <input type="password" id="password" value={password} onChange={(e) => setPassword(e.target.value)} className="text-primary font-mono font-bold p-2 rounded-lg"/>
                                <label className="text-2xl">password</label>
                            </div>
                        </div>
                    </div>
                    <div className="bg-primary text-white font-staat rounded-lg">
                        <button className="px-3 py-2 hover:bg-white hover:scale-125 hover:text-primary transition-all" type="submit">Login</button>
                    </div>
                </form>
            </div>    
        </div>
    )
}