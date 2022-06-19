import React from 'react';
import "../styles/loader.css";

const FullPageLoader = () => {
    return (
        <div className='spinner'>
            <div className="spinner-border" role="status">
                <span className="sr-only">Loading...</span>
            </div>
        </div>
        
    );
}


export default FullPageLoader;