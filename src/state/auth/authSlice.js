import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    "isLoggedIn": false,
    "Token": "",
    "email": "",
}

const authSlice = createSlice({
    name: "auth",
    initialState,
    reducers: {
        showAuthStat: (state) => {
            console.log(state["isLoggedIn"], state["Token"])
        },
        Login: (state, action) => {     
            window.localStorage.setItem('Token', action.payload);
            state["isLoggedIn"] = true;
            state["Token"] = action.payload;
        }
    }
})

export const { showAuthStat, Login } = authSlice.actions;

export default authSlice.reducer;